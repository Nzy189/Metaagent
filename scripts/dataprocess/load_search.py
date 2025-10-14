# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the search datasets to parquet format
Support multiple benchmarks: Bamboogle, 2Wiki, HotpotQA, Musique
Support both train and test split generation
"""

import re
import os
import datasets
import pandas as pd
import numpy as np

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']
    return question


def process_golden_answers(golden_answers, to_string=True):
    """
    Processes 'golden_answers' field and returns a STRING (comma-separated) or list.
    Handles: list, tuple, numpy array (1D or scalar), string, number, None, etc.
    """
    items = []

    # Case 1: numpy array
    if isinstance(golden_answers, np.ndarray):
        items = [str(item) for item in golden_answers.flatten() if item is not None and pd.notna(item)]
    # Case 2: list or tuple
    elif isinstance(golden_answers, (list, tuple)):
        items = [str(item) for item in golden_answers if item is not None and pd.notna(item)]
    # Case 3: string
    elif isinstance(golden_answers, str):
        cleaned = golden_answers.strip()
        if cleaned:
            items = [cleaned]
    # Case 4: scalar number (including np.number)
    elif isinstance(golden_answers, (int, float, np.generic)):
        if not pd.isna(golden_answers):
            items = [str(golden_answers).strip()]
    # Case 5: None or empty
    elif golden_answers is None or (isinstance(golden_answers, str) and not golden_answers.strip()):
        items = []
    # Fallback: try string conversion
    else:
        s = str(golden_answers).strip()
        if s and s != "nan":
            items = [s]

    if to_string:
        return "; ".join(items) if items else ""
    else:
        return items


def process_train_nq_dataset(dataset):
    """
    Processes the NQ train dataset to a unified schema for training.
    """
    processed_data = []
    for idx, item in enumerate(dataset):
        # Clean question
        question = item.get("question", "").strip()
        if question and not question.endswith('?'):
            question += '?'

        # Process golden_answers: convert to string
        golden_answers = item.get("golden_answers", [])
        final_result = process_golden_answers(golden_answers, to_string=True)

        # Build entry
        new_entry = {
            'id': idx,
            'question': question,
            'chain': "",
            'result': str(final_result),
            'source': "nq",
            'extra_info': {
                'ground_truth': str(final_result),
                'idx': idx
            }
        }
        processed_data.append(new_entry)

    df = pd.DataFrame(processed_data)
    return datasets.Dataset.from_pandas(df, preserve_index=False)


BENCHMARK_MAPPING = {
    'bamboogle': 'bamboogle',
    '2wiki': '2wikimultihopqa',
    'hotpotqa': 'hotpotqa',
    'musique': 'musique'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--benchmarks', type=str, default='bamboogle,2wiki,hotpotqa,musique',
                        help='Comma-separated list of benchmarks to process for test set')
    parser.add_argument('--generate_train', action='store_true',
                        help='Generate training set from NQ dataset')
    parser.add_argument('--generate_test', action='store_true',
                        help='Generate test sets from benchmarks')

    args = parser.parse_args()

    # If neither flag is set, generate both
    if not args.generate_train and not args.generate_test:
        args.generate_train = True
        args.generate_test = True

    # ===== Generate Training Set =====
    if args.generate_train:
        print("=" * 60)
        print("Generating Training Set from NQ")
        print("=" * 60)
        
        try:
            # Load NQ train dataset
            print("Loading NQ train dataset...")
            nq_train = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq', split='train')
            print(f"✅ Loaded {len(nq_train)} records from NQ train")
            
            # Process NQ train dataset
            print("Processing NQ train dataset...")
            processed_train = process_train_nq_dataset(nq_train)
            print(f"✅ Processed {len(processed_train)} records")
            
            # Shuffle the dataset
            print("Shuffling dataset...")
            shuffled_train = processed_train.shuffle(seed=42)
            
            # Re-index to ensure unique IDs
            print("Re-indexing dataset...")
            final_train = shuffled_train.map(lambda example, idx: {'id': idx}, with_indices=True)
            
            # Save to train directory
            train_dir = os.path.join(args.local_dir, 'train')
            os.makedirs(train_dir, exist_ok=True)
            output_path = os.path.join(train_dir, 'train.parquet')
            
            print(f"Saving to {output_path}...")
            final_train.to_parquet(output_path)
            print(f"✅ Saved {len(final_train)} records to {output_path}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to generate training set: {e}")
            import traceback
            traceback.print_exc()

    # ===== Generate Test Sets =====
    if args.generate_test:
        print("=" * 60)
        print("Generating Test Sets from Benchmarks")
        print("=" * 60)
        
        benchmark_list = [b.strip() for b in args.benchmarks.split(',')]
        
        test_dir = os.path.join(args.local_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)

        for benchmark_name in benchmark_list:
            if benchmark_name not in BENCHMARK_MAPPING:
                print(f"Warning: Unknown benchmark '{benchmark_name}', skipping...")
                continue
            
            print(f"Processing {benchmark_name}...")
            
            dataset_name = BENCHMARK_MAPPING[benchmark_name]
            
            try:
                # 加载数据集
                dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', dataset_name)
                
                # 获取测试集（如果没有test则使用dev或validation）
                if 'test' in dataset:
                    test_dataset = dataset['test']
                elif 'dev' in dataset:
                    test_dataset = dataset['dev']
                elif 'validation' in dataset:
                    test_dataset = dataset['validation']
                else:
                    print(f"Warning: No test/dev/validation split found for {benchmark_name}, skipping...")
                    continue
                
                # 定义数据处理函数
                def make_map_fn(data_source):
                    def process_fn(example, idx):
                        # 处理问题格式
                        question_text = example['question'].strip()
                        if question_text and question_text[-1] != '?':
                            question_text += '?'
                        example['question'] = question_text
                        
                        question = make_prefix(example, template_type=args.template_type)
                        solution = {
                            "target": example['golden_answers'],
                        }

                        data = {
                            "data_source": data_source,
                            "prompt": [{
                                "role": "user",
                                "content": question,
                            }],
                            "ability": "fact-reasoning",
                            "reward_model": {
                                "style": "rule",
                                "ground_truth": solution
                            },
                            "extra_info": {
                                'split': 'test',
                                'index': idx,
                            }
                        }
                        return data

                    return process_fn
           
                processed_dataset = test_dataset.map(
                    function=make_map_fn(benchmark_name), 
                    with_indices=True
                )
     
                output_file = os.path.join(test_dir, f'{benchmark_name}.parquet')
                processed_dataset.to_parquet(output_file)
                print(f"✅ Saved {benchmark_name} to {output_file}")
                
            except Exception as e:
                print(f"❌ Error processing {benchmark_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print()

    # ===== Upload to HDFS if specified =====
    if args.hdfs_dir is not None:
        print("=" * 60)
        print("Uploading to HDFS")
        print("=" * 60)
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print(f"✅ Uploaded to HDFS: {args.hdfs_dir}")
    
    print("\n" + "=" * 60)
    print("✅ All processing completed successfully!")
    print("=" * 60)
