#!/usr/bin/env python3
"""
Simple script to test the CodeTestAgentGraph with SGLang servers.

Usage:
    python run_test.py
    
Or with custom ports:
    python run_test.py --code-port 30000 --test-port 30001

Requirements:
    - SGLang servers running on specified ports
    - aiohttp package installed
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the rllm package to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rllm.agent_graphs.code.test_sglang_graph import CodeTestGraphTester


async def run_connectivity_test(code_port: int, test_port: int):
    """Test connectivity to SGLang servers."""
    print(f"Testing connectivity to SGLang servers...")
    print(f"Code server: localhost:{code_port}")
    print(f"Test server: localhost:{test_port}")
    
    tester = CodeTestGraphTester(
        code_agent_port=code_port,
        test_agent_port=test_port,
        max_steps=2  # Just for connectivity test
    )
    
    connectivity = await tester.test_connectivity()
    
    print("\nConnectivity Results:")
    print(f"  Code server (port {code_port}): {'✓ Connected' if connectivity['code_server'] else '✗ Failed'}")
    print(f"  Test server (port {test_port}): {'✓ Connected' if connectivity['test_server'] else '✗ Failed'}")
    
    return all(connectivity.values())


async def run_full_test(code_port: int, test_port: int, max_steps: int = 8):
    """Run full test with the graph."""
    print(f"\nRunning full CodeTestAgentGraph test...")
    
    tester = CodeTestGraphTester(
        code_agent_port=code_port,
        test_agent_port=test_port,
        max_steps=max_steps
    )
    
    # Run the simple test
    results = await tester.run_simple_test()
    
    # Print results
    tester.print_results(results)
    
    return results


async def run_custom_task(code_port: int, test_port: int, question: str, max_steps: int = 8):
    """Run test with custom programming question."""
    print(f"\nRunning test with custom question...")
    
    tester = CodeTestGraphTester(
        code_agent_port=code_port,
        test_agent_port=test_port,
        max_steps=max_steps
    )
    
    custom_task = {
        "question": question,
        "ground_truth": {
            "test_input": ["sample_input"],
            "test_output": ["sample_output"]
        }
    }
    
    results = await tester.run_task(custom_task)
    tester.print_results(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test CodeTestAgentGraph with SGLang servers")
    parser.add_argument("--code-port", type=int, default=30000, help="Port for code generation agent (default: 30000)")
    parser.add_argument("--test-port", type=int, default=30001, help="Port for test generation agent (default: 30001)")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum steps for the graph (default: 8)")
    parser.add_argument("--connectivity-only", action="store_true", help="Only test connectivity, don't run full test")
    parser.add_argument("--custom-question", type=str, help="Custom programming question to test")
    
    args = parser.parse_args()
    
    async def main_async():
        print("=" * 60)
        print("CodeTestAgentGraph SGLang Tester")
        print("=" * 60)
        
        # Test connectivity first
        connected = await run_connectivity_test(args.code_port, args.test_port)
        
        if not connected:
            print("\n⚠️  Warning: Some servers are not accessible!")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("Exiting...")
                return
        
        if args.connectivity_only:
            print("\nConnectivity test completed.")
            return
        
        try:
            if args.custom_question:
                # Run with custom question
                await run_custom_task(args.code_port, args.test_port, args.custom_question, args.max_steps)
            else:
                # Run default test
                await run_full_test(args.code_port, args.test_port, args.max_steps)
                
            print("\n✅ Test completed successfully!")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run the async main
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 