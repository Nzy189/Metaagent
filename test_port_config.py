#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_port_config.py - 测试端口配置功能

这个脚本用于测试修改后的端口配置是否正常工作
"""

import os
import sys
import subprocess
from pathlib import Path

def test_hydra_config_parsing():
    """测试 Hydra 配置解析是否能正确处理 vllm_address 参数"""
    print("🧪 测试 Hydra 配置解析...")
    
    # 模拟 Hydra 配置
    test_command = [
        "python3", "-c", 
        """
import sys
sys.path.append('/home/lah003/workspace/verl_efficient')
from omegaconf import OmegaConf

# 模拟通过命令行传入的配置
config = OmegaConf.create({
    'vllm_address': '127.0.0.1:8101',
    'models': {'model_0': {'path': '/test/path'}},
    'enable_thinking': False,
    'env': {'max_turns': 1},
    'benchmark': 'test'
})

# 测试地址解析逻辑
address = None
if hasattr(config, 'vllm_address') and config.vllm_address:
    address = config.vllm_address
    print(f'✅ 成功从配置中获取地址: {address}')
else:
    print('❌ 无法从配置中获取地址')
    
print(f'最终地址: {address}')
"""
    ]
    
    try:
        result = subprocess.run(test_command, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Hydra 配置解析测试通过")
            print(result.stdout)
        else:
            print("❌ Hydra 配置解析测试失败")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 测试执行错误: {e}")

def test_bash_parameter_parsing():
    """测试 bash 脚本参数解析"""
    print("\n🧪 测试 Bash 脚本参数解析...")
    
    # 测试脚本内容
    test_script_content = '''#!/bin/bash
# 测试参数解析
VLLM_ADDRESS=${1:-"127.0.0.1:8100"}
echo "传入参数: $1"
echo "解析后地址: $VLLM_ADDRESS"

# 测试不同情况
if [ -z "$1" ]; then
    echo "✅ 无参数时使用默认地址: $VLLM_ADDRESS"
else
    echo "✅ 有参数时使用指定地址: $VLLM_ADDRESS"
fi
'''
    
    # 创建临时测试脚本
    test_script_path = "/tmp/test_bash_params.sh"
    with open(test_script_path, "w") as f:
        f.write(test_script_content)
    
    os.chmod(test_script_path, 0o755)
    
    # 测试无参数情况
    print("测试无参数情况:")
    subprocess.run([test_script_path])
    
    # 测试有参数情况
    print("\n测试有参数情况:")
    subprocess.run([test_script_path, "192.168.1.100:8888"])
    
    # 清理临时文件
    os.unlink(test_script_path)

def test_environment_variable():
    """测试环境变量方式"""
    print("\n🧪 测试环境变量方式...")
    
    # 设置环境变量
    os.environ["VLLM_SERVICE_ADDRESS"] = "127.0.0.1:9999"
    
    test_command = [
        "python3", "-c", 
        """
import os
address = os.environ.get("VLLM_SERVICE_ADDRESS")
if address:
    print(f'✅ 成功从环境变量获取地址: {address}')
else:
    print('❌ 无法从环境变量获取地址')
"""
    ]
    
    result = subprocess.run(test_command, capture_output=True, text=True)
    print(result.stdout)
    
    # 清理环境变量
    del os.environ["VLLM_SERVICE_ADDRESS"]

def main():
    print("🚀 开始测试端口配置功能\n")
    
    test_hydra_config_parsing()
    test_bash_parameter_parsing() 
    test_environment_variable()
    
    print("\n📋 使用说明:")
    print("1. 通过命令行参数指定端口:")
    print("   ./validate_base.sh \"127.0.0.1:8101\"")
    print()
    print("2. 通过环境变量指定端口:")
    print("   export VLLM_SERVICE_ADDRESS=\"127.0.0.1:8102\"")
    print("   ./validate_base.sh")
    print()
    print("3. 使用默认端口:")
    print("   ./validate_base.sh")
    print()
    print("✅ 所有测试完成！")

if __name__ == "__main__":
    main()
