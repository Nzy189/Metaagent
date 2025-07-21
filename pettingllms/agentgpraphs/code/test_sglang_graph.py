import asyncio
import json
import logging
import aiohttp
from typing import Dict, List, Optional

from pettingllms.agent_graphs.code.code_test_graph import CodeTestAgentGraph

logger = logging.getLogger(__name__)


class SGLangClient:
    """
    Client for communicating with SGLang servers.
    """
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize SGLang client.
        
        Args:
            base_url: Base URL for SGLang server (e.g., "http://localhost:30000")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
    async def generate(self, messages: List[Dict], model: str = "default", **kwargs) -> str:
        """
        Generate response from SGLang server.
        
        Args:
            messages: List of messages in OpenAI chat format
            model: Model name
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            try:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        raise Exception(f"SGLang server error {response.status}: {error_text}")
            except asyncio.TimeoutError:
                raise Exception(f"Request timeout to {self.base_url}")
            except Exception as e:
                raise Exception(f"SGLang client error: {str(e)}")


class CodeTestGraphTester:
    """
    Tester for CodeTestAgentGraph using SGLang servers.
    """
    
    def __init__(self, 
                 code_agent_port: int = 30000,
                 test_agent_port: int = 30001,
                 host: str = "localhost",
                 max_steps: int = 10,
                 timeout: int = 30):
        """
        Initialize the tester.
        
        Args:
            code_agent_port: Port for code generation agent's SGLang server
            test_agent_port: Port for test generation agent's SGLang server  
            host: Host address
            max_steps: Maximum steps for the graph
            timeout: Request timeout
        """
        self.code_client = SGLangClient(f"http://{host}:{code_agent_port}", timeout)
        self.test_client = SGLangClient(f"http://{host}:{test_agent_port}", timeout)
        
        # Initialize the graph
        self.graph = CodeTestAgentGraph(max_steps=max_steps)
        
        logger.info(f"Initialized tester with code port {code_agent_port} and test port {test_agent_port}")
    
    async def test_connectivity(self) -> Dict[str, bool]:
        """
        Test connectivity to both SGLang servers.
        
        Returns:
            Dictionary with connectivity status for both servers
        """
        results = {
            "code_server": False,
            "test_server": False
        }
        
        test_messages = [{"role": "user", "content": "Hello, can you respond?"}]
        
        # Test code server
        try:
            response = await self.code_client.generate(test_messages)
            if response:
                results["code_server"] = True
                logger.info("Code server (port 30000) is accessible")
        except Exception as e:
            logger.error(f"Code server connectivity failed: {str(e)}")
        
        # Test test server
        try:
            response = await self.test_client.generate(test_messages)
            if response:
                results["test_server"] = True
                logger.info("Test server (port 30001) is accessible")
        except Exception as e:
            logger.error(f"Test server connectivity failed: {str(e)}")
        
        return results
    
    async def run_task(self, task: Dict) -> Dict:
        """
        Run a complete task through the graph using SGLang servers.
        
        Args:
            task: Task dictionary with programming problem
            
        Returns:
            Complete execution results
        """
        logger.info(f"Starting task: {task.get('question', 'Unknown')[:50]}...")
        
        # Reset graph and get initial observation
        initial_obs = self.graph.reset(task)
        
        execution_log = []
        step_count = 0
        
        while not self.graph.is_done and step_count < self.graph.max_steps:
            try:
                # Get current agent's chat history
                chat_history = self.graph.get_next_agent_chat_completions()
                current_agent = self.graph.current_agent
                
                logger.info(f"Step {step_count + 1}: Getting response from {self.graph.current_agent_name}")
                
                # Select appropriate client based on current agent
                if current_agent == "code":
                    client = self.code_client
                else:
                    client = self.test_client
                
                # Generate response from appropriate SGLang server
                model_response = await client.generate(
                    messages=chat_history,
                    max_tokens=2048,
                    temperature=0.1
                )
                
                # Execute step in graph
                next_obs, reward, done, info = self.graph.step(model_response)
                
                # Log execution details
                execution_log.append({
                    "step": step_count + 1,
                    "agent": current_agent,
                    "agent_name": self.graph.current_agent_name,
                    "model_response": model_response[:200] + "..." if len(model_response) > 200 else model_response,
                    "reward": reward,
                    "done": done,
                    "observation_keys": list(next_obs.keys()) if isinstance(next_obs, dict) else "non-dict",
                    "info": info
                })
                
                logger.info(f"Step {step_count + 1} completed. Reward: {reward}, Done: {done}")
                
                step_count += 1
                
                if done:
                    logger.info("Task completed by environment")
                    break
                    
            except Exception as e:
                logger.error(f"Error in step {step_count + 1}: {str(e)}")
                execution_log.append({
                    "step": step_count + 1,
                    "agent": self.graph.current_agent,
                    "error": str(e)
                })
                break
        
        # Get final results
        final_results = {
            "task": task,
            "execution_log": execution_log,
            "total_steps": step_count,
            "total_reward": self.graph.total_reward,
            "completed": self.graph.is_done,
            "graph_state": self.graph.get_current_state(),
            "all_trajectories": self.graph.get_all_trajectories()
        }
        
        logger.info(f"Task completed. Total steps: {step_count}, Total reward: {self.graph.total_reward}")
        return final_results

    async def run_simple_test(self) -> Dict:
        """
        Run a simple test with a basic programming problem.
        
        Returns:
            Test execution results
        """
        test_task = {
            "question": """
Write a Python function that takes a list of integers and returns the sum of all even numbers in the list.

Example:
Input: [1, 2, 3, 4, 5, 6]
Output: 12 (2 + 4 + 6)

Function signature: def sum_even_numbers(numbers):
            """.strip(),
            "ground_truth": {
                "test_input": [
                    "[1, 2, 3, 4, 5, 6]",
                    "[10, 15, 20, 25]", 
                    "[]",
                    "[1, 3, 5]"
                ],
                "test_output": [
                    "12",
                    "30",
                    "0", 
                    "0"
                ]
            }
        }
        
        return await self.run_task(test_task)

    def print_results(self, results: Dict):
        """
        Print formatted results of task execution.
        
        Args:
            results: Results dictionary from run_task
        """
        print("\n" + "="*80)
        print("CODE TEST GRAPH EXECUTION RESULTS")
        print("="*80)
        
        print(f"Task: {results['task']['question'][:100]}...")
        print(f"Total Steps: {results['total_steps']}")
        print(f"Total Reward: {results['total_reward']}")
        print(f"Completed: {results['completed']}")
        print()
        
        print("EXECUTION LOG:")
        print("-"*40)
        for log_entry in results['execution_log']:
            if 'error' in log_entry:
                print(f"Step {log_entry['step']}: ERROR - {log_entry['error']}")
            else:
                print(f"Step {log_entry['step']}: {log_entry['agent_name']}")
                print(f"  Reward: {log_entry['reward']}")
                print(f"  Response: {log_entry['model_response']}")
                print()
        
        graph_state = results['graph_state']
        print("FINAL GRAPH STATE:")
        print("-"*40)
        print(f"Environment state: {graph_state['env_state']}")
        print()


async def main():
    """
    Main function to run the SGLang graph test.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize tester
    tester = CodeTestGraphTester(
        code_agent_port=30000,
        test_agent_port=30001,
        max_steps=8
    )
    
    # Test connectivity
    print("Testing SGLang server connectivity...")
    connectivity = await tester.test_connectivity()
    print(f"Code server (30000): {'✓' if connectivity['code_server'] else '✗'}")
    print(f"Test server (30001): {'✓' if connectivity['test_server'] else '✗'}")
    
    if not all(connectivity.values()):
        print("Warning: Some servers are not accessible. Proceeding anyway...")
    
    # Run simple test
    print("\nRunning simple test task...")
    try:
        results = await tester.run_simple_test()
        tester.print_results(results)
        
        # Save results to file
        with open("sglang_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("\nResults saved to sglang_test_results.json")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        logger.exception("Test execution failed")


if __name__ == "__main__":
    asyncio.run(main()) 