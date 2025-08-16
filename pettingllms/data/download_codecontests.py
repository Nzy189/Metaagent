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
Download Gen-Verse/CodeContests_train dataset from Hugging Face to local directory
"""

import os
import datasets
import argparse
from pathlib import Path

def download_codecontests_dataset(local_dir: str = None, name: str = "CodeContests_train", split: str = "train"):
    """
    Download Gen-Verse/CodeContests_train dataset from Hugging Face
    
    Args:
        local_dir: Local directory to save the dataset. Defaults to datasets/codecontests in root directory
        name: Dataset name to download. Defaults to "CodeContests_train"
        split: Dataset split to download. Defaults to "train"
    """
    if local_dir is None or local_dir == "None":
        # Default to root directory datasets/codecontests
        current_dir = Path(__file__).parent
        root_dir = current_dir.parent.parent  # Go up to root directory
        local_dir = root_dir / "datasets" / name
    
    local_dir = Path(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Gen-Verse/{name} dataset...")
    print(f"Local directory: {local_dir}")
    
    try:
        # Load dataset from Hugging Face
        dataset = datasets.load_dataset(f"Gen-Verse/{name}", split=split)
        
        print(f"Dataset loaded successfully!")
        print(f"Dataset size: {len(dataset)}")
        print(f"Dataset features: {dataset.features}")
        
        # Save to parquet format
        output_file = local_dir / f"{split}.parquet"
        dataset.to_parquet(output_file)
        print(f"Dataset saved to: {output_file}")
        
        # Also save as JSON for easier inspection
        json_file = local_dir / f"{split}.json"
        dataset.to_json(json_file)
        print(f"Dataset also saved as JSON to: {json_file}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument('--local_dir', type=str, default=None, 
                       help='Local directory to save the dataset')
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'validation', 'test'],
                       help='Dataset split to download')
    parser.add_argument('--name', type=str, default='CodeContests_train', 
                       help='Dataset name to download')
    
    args = parser.parse_args()
    
    success = download_codecontests_dataset(
        local_dir=args.local_dir,
        split=args.split,
        name=args.name
    )
    
    if success:
        print("Dataset download completed successfully!")
    else:
        print("Dataset download failed!")
        exit(1)

if __name__ == '__main__':
    main()
