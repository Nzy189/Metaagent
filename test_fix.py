#!/usr/bin/env python3
"""
测试修复后的 math_utils.py 中的函数，确保它们能正确处理各种输入类型。
"""

import sys
import os
sys.path.append('/home/lah003/workspace/PettingLLMs')

from pettingllms.multi_agent_env.math.math_utils import _both_numbers, _num_equal, _simplify_equal, _equals_random_samples, symbolic_equal
import sympy as sp

def test_both_numbers():
    """测试 _both_numbers 函数"""
    print("测试 _both_numbers 函数...")
    
    # 正常情况
    assert _both_numbers(sp.Integer(1), sp.Integer(2)) == True
    assert _both_numbers(sp.Float(1.5), sp.Integer(2)) == True
    
    # 异常情况 - 列表
    assert _both_numbers([1, 2], sp.Integer(2)) == False
    assert _both_numbers(sp.Integer(1), [1, 2]) == False
    assert _both_numbers([1, 2], [3, 4]) == False
    
    # 异常情况 - None
    assert _both_numbers(None, sp.Integer(2)) == False
    assert _both_numbers(sp.Integer(1), None) == False
    assert _both_numbers(None, None) == False
    
    # 异常情况 - 其他类型
    assert _both_numbers("string", sp.Integer(2)) == False
    assert _both_numbers(sp.Integer(1), "string") == False
    
    print("✓ _both_numbers 测试通过")

def test_num_equal():
    """测试 _num_equal 函数"""
    print("测试 _num_equal 函数...")
    
    # 正常情况
    assert _num_equal(sp.Integer(1), sp.Integer(1)) == True
    assert _num_equal(sp.Float(1.0), sp.Integer(1)) == True
    
    # 异常情况
    assert _num_equal([1, 2], sp.Integer(1)) == False
    assert _num_equal(sp.Integer(1), [1, 2]) == False
    assert _num_equal(None, sp.Integer(1)) == False
    assert _num_equal("string", sp.Integer(1)) == False
    
    print("✓ _num_equal 测试通过")

def test_simplify_equal():
    """测试 _simplify_equal 函数"""
    print("测试 _simplify_equal 函数...")
    
    # 正常情况
    x = sp.Symbol('x')
    assert _simplify_equal(x + 1, x + 1) == True
    
    # 异常情况
    assert _simplify_equal([1, 2], x + 1) == False
    assert _simplify_equal(x + 1, [1, 2]) == False
    assert _simplify_equal(None, x + 1) == False
    assert _simplify_equal("string", x + 1) == False
    
    print("✓ _simplify_equal 测试通过")

def test_equals_random_samples():
    """测试 _equals_random_samples 函数"""
    print("测试 _equals_random_samples 函数...")
    
    # 正常情况
    x = sp.Symbol('x')
    assert _equals_random_samples(x + 1, x + 1) == True
    
    # 异常情况
    assert _equals_random_samples([1, 2], x + 1) == False
    assert _equals_random_samples(x + 1, [1, 2]) == False
    assert _equals_random_samples(None, x + 1) == False
    assert _equals_random_samples("string", x + 1) == False
    
    print("✓ _equals_random_samples 测试通过")

def test_symbolic_equal():
    """测试 symbolic_equal 函数"""
    print("测试 symbolic_equal 函数...")
    
    # 正常情况
    assert symbolic_equal("1", "1") == True
    assert symbolic_equal("2/3", "0.6666666666666666") == True
    
    # 测试可能导致原始错误的情况
    # 这些应该不会抛出异常
    try:
        result = symbolic_equal("", "1")
        print(f"空字符串测试结果: {result}")
        
        result = symbolic_equal("invalid_math", "1")
        print(f"无效数学表达式测试结果: {result}")
        
        print("✓ symbolic_equal 测试通过")
    except Exception as e:
        print(f"✗ symbolic_equal 测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("开始测试修复后的 math_utils.py 函数...")
    print("=" * 50)
    
    try:
        test_both_numbers()
        test_num_equal()
        test_simplify_equal()
        test_equals_random_samples()
        test_symbolic_equal()
        
        print("=" * 50)
        print("✓ 所有测试都通过了！修复成功。")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
