#!/usr/bin/env python3
"""
简单处理日志文件的脚本
将日志读取为dict列表，并计算termination_reason为all_tests_passed的比例
"""

import json
import re

def process_log_file(log_file_path):
    """
    处理日志文件，返回dict列表和统计信息
    """
    rollouts = []
    
    # 读取文件内容
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式分割每个rollout条目
    # 匹配模式：[timestamp] [ROLLOUT:number] { ... }
    pattern = r'\[([^\]]+)\] \[ROLLOUT:(\d+)\] (\{.*?\n\})'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        timestamp_str, rollout_idx, json_str = match
        try:
            # 解析JSON
            rollout_data = json.loads(json_str)
            rollout_data['log_timestamp'] = timestamp_str
            rollouts.append(rollout_data)
        except json.JSONDecodeError as e:
            print(f"跳过无法解析的JSON (rollout {rollout_idx}): {e}")
            continue
    
    return rollouts

def calculate_statistics(rollouts):
    """
    计算termination_reason的统计信息
    """
    total_count = len(rollouts)
    all_tests_passed_count = 0
    termination_reasons = {}
    finally_pass_cases=[]
    finally_test_passed_cases=[]
    init_pass_cases=[]
    init_test_passed_cases=[]
    success_generated_code_count=0
    for idx,rollout in enumerate(rollouts):
        reason = rollout.get('termination_reason', 'unknown')
        agent_rewards=rollout.get('agent_rewards', {})
        if agent_rewards:
            print("extracted agent_rewards")
            code_generator=agent_rewards.get('code_generator', '')
            test_generator=agent_rewards.get('test_generator', '')
            if code_generator:
                if code_generator==1:
                    finally_pass_cases.append(rollout)
                    if test_generator==1:
                        finally_test_passed_cases.append(rollout)
       
        
        extra_data = rollout.get('extra_data', {})
        if extra_data:
            
            env_state = extra_data.get('env_state', {})
            if env_state:
                generated_code=env_state.get('generated_code', '')
                if len(generated_code)>0:
                    success_generated_code_count+=1
           
            reward_history_dict=extra_data.get('reward_history_dict', {})
            if reward_history_dict:
                print("extracted reward_history_dict")
                code_generator=reward_history_dict.get('code_generator', [])
                test_generator=reward_history_dict.get('test_generator', [])
                if code_generator!=[]:
                    if code_generator[0]==1:
                        init_pass_cases.append(rollout)
                        if test_generator[0]==1:
                            init_test_passed_cases.append(rollout)
                       
        if reason == 'all_tests_passed':
            all_tests_passed_count += 1
    
    # 计算比例
    #total_count=508
    all_tests_passed_ratio = all_tests_passed_count / total_count if total_count > 0 else 0
    finally_pass_cases_ratio = len(finally_pass_cases) / total_count if total_count > 0 else 0
    init_pass_cases_ratio = len(init_pass_cases) / total_count if total_count > 0 else 0
    init_pass_cases_ratio_not_in_finally_pass_cases=[]
    finally_pass_cases_ratio_not_in_init_pass_cases=[]
    for a in init_pass_cases:
        if a not in finally_pass_cases:
            init_pass_cases_ratio_not_in_finally_pass_cases.append(a)
    for a in finally_pass_cases:
        if a not in init_pass_cases:
            finally_pass_cases_ratio_not_in_init_pass_cases.append(a)
    print("init_pass_cases_ratio_not_in_finally_pass_cases:")
    for a in init_pass_cases_ratio_not_in_finally_pass_cases:
        print(a['extra_data']['reward_history_dict'])
    print("finally_pass_cases_ratio_not_in_init_pass_cases:")
    for a in finally_pass_cases_ratio_not_in_init_pass_cases:
        print(a['extra_data']['reward_history_dict'])
    return {
        'total_count': total_count,
        'all_tests_passed_count': all_tests_passed_count,
        'all_tests_passed_ratio': all_tests_passed_ratio,
        'termination_reasons': termination_reasons,
        'success_generated_code_count': success_generated_code_count/total_count,
        'finally_pass_cases_ratio': finally_pass_cases_ratio,
        'init_pass_cases_ratio': init_pass_cases_ratio,
        'init_pass_cases_ratio_not_in_finally_pass_ratio': len(init_pass_cases_ratio_not_in_finally_pass_cases)/total_count,
        'finally_pass_cases_ratio_not_in_init_pass_ratio': len(finally_pass_cases_ratio_not_in_init_pass_cases)/total_count,
        'finally_test_passed_cases_ratio': len(finally_test_passed_cases)/total_count,
        'init_test_passed_cases_ratio': len(init_test_passed_cases)/total_count
    }

def main():
    # 日志文件路径
    #log_file = 'logs/2025-08-18/MBPP/summary.log'
    #log_file = 'logs/2025-08-18/Live_code_bench/summary.log'
    #log_file = 'logs/2025-08-18/Livebench/summary.log'
    log_file = 'logs/2025-08-18/Code_forces/summary.log'
    print("正在处理日志文件...")
    
    # 处理日志文件
    rollouts = process_log_file(log_file)
    
    print(f"成功读取 {len(rollouts)} 个rollout条目")
    
    # 计算统计信息
    stats = calculate_statistics(rollouts)
    
    # 打印结果
    print("\n=== 统计结果 ===")
    print(f"总条目数: {stats['total_count']}")
    print(f"all_tests_passed 条目数: {stats['all_tests_passed_count']}")
    print(f"all_tests_passed 比例: {stats['all_tests_passed_ratio']:.4f} ({stats['all_tests_passed_ratio']*100:.2f}%)")
    print(f"finally_pass_cases 比例: {stats['finally_pass_cases_ratio']:.4f} ({stats['finally_pass_cases_ratio']*100:.2f}%)")
    print(f"init_pass_cases 比例: {stats['init_pass_cases_ratio']:.4f} ({stats['init_pass_cases_ratio']*100:.2f}%)")
    print(f"对的改错比例: {stats['init_pass_cases_ratio_not_in_finally_pass_ratio']:.4f} ({stats['init_pass_cases_ratio_not_in_finally_pass_ratio']*100:.2f}%)")
    print(f"错的改对比例: {stats['finally_pass_cases_ratio_not_in_init_pass_ratio']:.4f} ({stats['finally_pass_cases_ratio_not_in_init_pass_ratio']*100:.2f}%)")
    print(f"finally_test_passed_cases 比例: {stats['finally_test_passed_cases_ratio']:.4f} ({stats['finally_test_passed_cases_ratio']*100:.2f}%)")
    print(f"init_test_passed_cases 比例: {stats['init_test_passed_cases_ratio']:.4f} ({stats['init_test_passed_cases_ratio']*100:.2f}%)")
    #print(f"success_generated_code_count 比例: {stats['success_generated_code_count']:.4f} ({stats['success_generated_code_count']*100:.2f}%)")
    
  
    
    # 返回数据供进一步使用
    return rollouts, stats

if __name__ == '__main__':
    rollouts, stats = main()
    