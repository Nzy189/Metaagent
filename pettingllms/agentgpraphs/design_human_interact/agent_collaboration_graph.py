"""
Frontend Design Task Dual-Agent Collaboration Test System

This system implements agent testing using HuggingFaceM4/WebSight dataset samples
and communicates with two sglang ports for different agents, following sweet_rl pattern.

Key Features:
1. Load test samples from HuggingFaceM4/WebSight dataset
2. Use sglang servers on different ports for agents
3. Implement sweet_rl style evaluation workflow
4. Generate evaluation metrics and results
"""

import os
import sys
import json
import tempfile
import requests
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from agents.visual_info_agent import VisualInfoAgent
from agents.code_genaration_agent import CodeGenerationAgent
from websight_env import WebEnv

# Import sweet_rl rendering tools
try:
    from sweet_rl.sweet_rl.utils.webpage_utils import (
        render_full_html, get_driver, replace_urls
    )
    SWEET_RL_AVAILABLE = True
except ImportError:
    print("Warning: sweet_rl rendering tools not available")
    SWEET_RL_AVAILABLE = False


class SGLangAgentClient:
    """
    Client for communicating with sglang agents on different ports
    """
    
    def __init__(self, hostname: str = "localhost", code_port: int = 8000, visual_port: int = 8001):
        """
        Initialize sglang client
        
        Args:
            hostname: Server hostname
            code_port: Port for code generation agent
            visual_port: Port for visual analysis agent
        """
        self.hostname = hostname
        self.code_port = code_port
        self.visual_port = visual_port
        self.code_base_url = f"http://{hostname}:{code_port}/v1"
        self.visual_base_url = f"http://{hostname}:{visual_port}/v1"
        
    def _make_request(self, base_url: str, messages: List[Dict], max_tokens: int = 4096) -> str:
        """
        Make request to sglang server
        
        Args:
            base_url: Base URL for the server
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            payload = {
                "model": "default",  # sglang uses "default" model name
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }
            
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"❌ Request failed with status {response.status_code}: {response.text}")
                return f"Error: Failed to get response from server"
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            return f"Error: {str(e)}"
    
    def get_code_response(self, messages: List[Dict]) -> str:
        """Get response from code generation agent"""
        return self._make_request(self.code_base_url, messages, max_tokens=8192)
    
    def get_visual_response(self, messages: List[Dict]) -> str:
        """Get response from visual analysis agent"""
        return self._make_request(self.visual_base_url, messages, max_tokens=4096)
    
    def test_connections(self) -> bool:
        """Test connections to both servers"""
        print("🔗 Testing connections to sglang servers...")
        
        test_messages = [{"role": "user", "content": "Hello, please respond with 'OK'"}]
        
        # Test code generation server
        try:
            code_response = self.get_code_response(test_messages)
            print(f"✅ Code generation server ({self.code_port}): Connected")
        except Exception as e:
            print(f"❌ Code generation server ({self.code_port}): {e}")
            return False
            
        # Test visual analysis server
        try:
            visual_response = self.get_visual_response(test_messages)
            print(f"✅ Visual analysis server ({self.visual_port}): Connected")
        except Exception as e:
            print(f"❌ Visual analysis server ({self.visual_port}): {e}")
            return False
            
        return True


class FrontendDesignAgentGraph:
    """
    Frontend design test system using WebSight dataset and sglang servers
    """
    
    def __init__(
        self, 
        hostname: str = "localhost",
        code_port: int = 8000,
        visual_port: int = 8001,
        max_iterations: int = 3,
        temp_path: Optional[str] = None
    ):
        """
        Initialize test system
        
        Args:
            hostname: Server hostname
            code_port: Port for code generation agent
            visual_port: Port for visual analysis agent
            max_iterations: Maximum iterations per task
            temp_path: Temporary file path
        """
        self.client = SGLangAgentClient(hostname, code_port, visual_port)
        self.visual_agent = VisualInfoAgent()
        self.code_agent = CodeGenerationAgent(max_iterations=max_iterations)
        self.max_iterations = max_iterations
        self.temp_path = temp_path or tempfile.gettempdir()
        
        # Initialize webdriver if sweet_rl is available
        self.driver = None
        if SWEET_RL_AVAILABLE:
            try:
                self.driver = get_driver()
                print("✅ WebDriver initialized successfully")
            except Exception as e:
                print(f"⚠️ WebDriver initialization failed: {e}")
                self.driver = None
        
    def render_html_to_image(self, html_code: str, task_id: str, iteration: int = 0) -> Optional[str]:
        """
        Render HTML code to image using sweet_rl tools
        
        Args:
            html_code: HTML code to render
            task_id: Task identifier
            iteration: Iteration number
            
        Returns:
            Path to rendered image file
        """
        if not self.driver or not SWEET_RL_AVAILABLE:
            print("⚠️ WebDriver not available, using mock rendering")
            # Create mock image path for testing
            image_path = os.path.join(self.temp_path, f"{task_id}_iter_{iteration}.png")
            return image_path
            
        try:
            # Convert task_id to hash for env_id integer requirement
            env_id = hash(f"{task_id}_{iteration}") & 0x7FFFFFFF  # Ensure positive int
            image_path = render_full_html(
                self.driver, 
                html_code, 
                self.temp_path, 
                env_id=env_id
            )
            return image_path
        except Exception as e:
            print(f"❌ Rendering failed: {e}")
            return None
    
    def render_ground_truth(self, ground_truth_html: str, task_id: str) -> Optional[str]:
        """
        Render ground truth HTML to image
        
        Args:
            ground_truth_html: Ground truth HTML code
            task_id: Task identifier
            
        Returns:
            Path to ground truth image file
        """
        return self.render_html_to_image(ground_truth_html, f"{task_id}_gt", 0)
    
    def _get_agents_list(self) -> List[Tuple[str, Any, str]]:
        """
        Automatically detect agents and return them as a list with standardized names
        
        Returns:
            List of (agent_name, agent_instance, original_name) tuples
        """
        agents = []
        
        # Dynamically find all agent attributes
        agent_attrs = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if attr_name.endswith('_agent') and hasattr(attr, 'reset') and hasattr(attr, 'update_from_env'):
                agent_attrs.append((attr_name, attr))
        
        # Sort for consistent ordering
        agent_attrs.sort(key=lambda x: x[0])
        
        # Create standardized names: agent1, agent2, etc.
        for i, (original_name, agent_instance) in enumerate(agent_attrs, 1):
            standardized_name = f"agent{i}"
            agents.append((standardized_name, agent_instance, original_name))
        
        return agents
    
    def _get_agent_client_method(self, original_agent_name: str):
        """
        Get the appropriate client method for an agent based on its original name
        
        Args:
            original_agent_name: Original name of the agent (e.g., 'code_agent', 'visual_agent')
            
        Returns:
            Client method to use for this agent
        """
        if 'code' in original_agent_name.lower():
            return self.client.get_code_response
        elif 'visual' in original_agent_name.lower():
            return self.client.get_visual_response
        else:
            # Default to code response for unknown agent types
            return self.client.get_code_response
    
    async def loop(
        self, 
        observation: Dict,
        step_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Single-step multi-agent interaction loop
        
        This function handles one iteration of agent-environment interaction,
        focusing only on essential data updates: prompt, response, action, reward.
        
        Args:
            observation: Current observation from environment
            step_idx: Current step index
            
        Returns:
            Dictionary containing essential update data for all agents
        """
        # Automatically detect agents
        agents_info = self._get_agents_list()
        
        # Initialize step update data
        step_update_data = {}
        
        # Process each agent in sequence
        for agent_name, agent_instance, original_name in agents_info:
            # ============ AGENT PROCESSING ============
            # Update agent with environment observation
            agent_instance.update_from_env(observation, reward=0.0, done=False, info={})
            
            # Get prompt messages
            prompt_messages = agent_instance.chat_completions.copy()
            
            # Get appropriate client method for this agent
            client_method = self._get_agent_client_method(original_name)
            
            # Get model response
            response = client_method(prompt_messages)
            
            # Update agent with model response
            action = agent_instance.update_from_model(response)
            
            # Get agent-specific action data
            if hasattr(agent_instance, 'get_generated_html'):
                action_data = agent_instance.get_generated_html()
                action_type = "code"
            elif hasattr(agent_instance, 'get_suggestions_summary'):
                action_data = agent_instance.get_suggestions_summary()
                action_type = "visual"
            else:
                action_data = str(action.action) if hasattr(action, 'action') else str(action)
                action_type = "generic"
            
            # Store essential agent data
            step_update_data[agent_name] = {
                "original_name": original_name,
                "prompt": prompt_messages,
                "response": response,
                "action": action_data,
                "action_type": action_type,
                "reward": 0.0  # Will be updated after environment step
            }
        
        return step_update_data
    
    async def update_rewards(
        self,
        step_update_data: Dict[str, Any],
        env_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update agent rewards after environment step
        
        Args:
            step_update_data: Data from loop() function
            env_results: Results from environment step containing rewards
            
        Returns:
            Updated step data with rewards
        """
        # Update rewards for each agent
        for agent_name in step_update_data:
            if agent_name in env_results.get('rewards', {}):
                step_update_data[agent_name]["reward"] = env_results['rewards'][agent_name]
            else:
                # Default reward if not specified
                step_update_data[agent_name]["reward"] = env_results.get('default_reward', 0.0)
        
        return step_update_data
    
    def evaluate_single_task(self, sample: Dict) -> Dict:
        """
        Evaluate a single task sample using the new simplified loop function
        
        Args:
            sample: Sample from WebSight dataset
            
        Returns:
            Evaluation results
        """
        task_id = sample["task_id"]
        print(f"\n🎯 Evaluating task: {task_id}")
        print(f"📝 Description: {sample['problem_description'][:100]}...")
        
        # Initialize environment
        env = WebEnv(task=sample, max_turns=self.max_iterations * 2, temp_path=self.temp_path)
        
        # Reset all agents
        agents_info = self._get_agents_list()
        for _, agent_instance, _ in agents_info:
            agent_instance.reset()
        
        # Get initial observation
        obs, _ = env.reset()
        
        # Initialize results structure
        results = {
            "task_id": task_id,
            "task_description": sample["problem_description"],
            "ground_truth_html": sample["ground_truth"],
            "reference_image": obs["reference_image"],
            "success": False,
            "total_iterations": 0,
            "final_html": "",
            "termination_reason": "",
            "iterations": [],
            "agent_data": {}  # Store cumulative agent data
        }
        
        # Initialize agent data storage
        for agent_name, _, original_name in agents_info:
            results["agent_data"][agent_name] = {
                "original_name": original_name,
                "steps": [],
                "total_reward": 0.0
            }
        
        # Multi-step interaction loop (handled in evaluate_single_task)
        try:
            for iteration in range(self.max_iterations):
                print(f"  🔄 Iteration {iteration + 1}/{self.max_iterations}")
                
                # Run single-step loop to get agent actions
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                step_update_data = loop.run_until_complete(
                    self.loop(obs, step_idx=iteration)
                )
                loop.close()
                
                # Process each agent's action with environment
                iteration_result = {
                    "iteration": iteration + 1,
                    "agent_actions": {},
                    "current_image": None,
                    "visual_suggestions": "",
                    "updated_html": "",
                    "visual_response": "",
                    "code_response": ""
                }
                
                for agent_name, agent_data in step_update_data.items():
                    action_type = agent_data["action_type"]
                    action_data_value = agent_data["action"]
                    
                    # Take step in environment
                    obs, reward, done, info = env.step(action_type, action_data_value)
                    
                    # Update rewards
                    env_results = {"default_reward": reward}
                    step_update_data = asyncio.run(
                        self.update_rewards(step_update_data, env_results)
                    )
                    
                    # Store agent step data
                    agent_step = {
                        "prompt": agent_data["prompt"],
                        "response": agent_data["response"],
                        "action": agent_data["action"],
                        "action_type": action_type,
                        "reward": reward
                    }
                    results["agent_data"][agent_name]["steps"].append(agent_step)
                    results["agent_data"][agent_name]["total_reward"] += reward
                    
                    # Store action data for backward compatibility
                    iteration_result["agent_actions"][agent_name] = agent_data
                    
                    # Extract data for backward compatibility
                    if action_type == "code":
                        iteration_result["updated_html"] = action_data_value
                        iteration_result["code_response"] = agent_data["response"]
                    elif action_type == "visual":
                        iteration_result["visual_suggestions"] = action_data_value
                        iteration_result["visual_response"] = agent_data["response"]
                
                # Update current image
                iteration_result["current_image"] = obs.get("current_image")
                
                # Check termination conditions
                if not obs.get("current_image"):
                    print("  ❌ HTML rendering failed")
                    results["termination_reason"] = "RENDERING_FAILED"
                    break
                
                # Check if design is satisfactory (using visual response)
                visual_response = iteration_result.get("visual_response", "")
                if self._is_design_satisfactory(visual_response):
                    print("  ✨ Design achieved satisfactory results!")
                    results["success"] = True
                    results["termination_reason"] = "TASK_COMPLETED"
                    break
                
                # Store iteration results
                results["iterations"].append(iteration_result)
                
                if done:
                    results["termination_reason"] = "ENV_DONE"
                    break
        
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            results["termination_reason"] = f"ERROR: {str(e)}"
        
        # Finalize results
        results["total_iterations"] = len(results["iterations"])
        
        # Get final HTML if available
        try:
            if hasattr(self, 'code_agent') and hasattr(self.code_agent, 'get_generated_html'):
                results["final_html"] = self.code_agent.get_generated_html()
        except:
            results["final_html"] = ""
        
        # Cleanup
        env.cleanup()
        
        return results
    
    def _is_design_satisfactory(self, visual_response: str) -> bool:
        """Check if design is satisfactory"""
        satisfactory_keywords = [
            "perfect match", "very close", "already achieved", "perfectly matches",
            "no modifications needed", "design complete", "extremely close"
        ]
        
        response_lower = visual_response.lower()
        return any(keyword in response_lower for keyword in satisfactory_keywords)
    
    def run_evaluation(
        self, 
        num_samples: int = 10, 
        output_path: Optional[str] = None,
        dataset_version: str = "v0.2"
    ) -> Dict:
        """
        Run evaluation on WebSight dataset samples
        
        Args:
            num_samples: Number of samples to evaluate
            output_path: Path to save results
            dataset_version: WebSight dataset version
            
        Returns:
            Evaluation summary
        """
        print(f"🚀 Starting Frontend Design Agent Evaluation")
        print(f"📊 Samples: {num_samples}, Max iterations: {self.max_iterations}")
        
        # Test connections
        if not self.client.test_connections():
            return {"error": "Failed to connect to sglang servers"}
        
        # Create environment and load dataset
        env = WebEnv(temp_path=self.temp_path)
        samples = env.load_dataset(num_samples, dataset_version)
        
        if not samples:
            return {"error": "Failed to load dataset samples"}
        
        # Run evaluation
        all_results = []
        success_count = 0
        
        for i, sample in enumerate(tqdm(samples, desc="Evaluating tasks")):
            try:
                result = self.evaluate_single_task(sample)
                all_results.append(result)
                
                if result.get("success", False):
                    success_count += 1
                    
                print(f"  ✅ Task {i+1}/{len(samples)} completed. Success: {result.get('success', False)}")
                
            except Exception as e:
                print(f"  ❌ Task {i+1} failed: {e}")
                all_results.append({
                    "task_id": sample.get("task_id", f"task_{i}"),
                    "error": str(e)
                })
        
        # Calculate metrics
        total_tasks = len(samples)
        success_rate = success_count / total_tasks if total_tasks > 0 else 0
        avg_iterations = sum(r.get("total_iterations", 0) for r in all_results) / total_tasks
        
        summary = {
            "total_tasks": total_tasks,
            "successful_tasks": success_count,
            "success_rate": success_rate,
            "average_iterations": avg_iterations,
            "detailed_results": all_results
        }
        
        # Save results
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"💾 Results saved to: {output_path}")
        
        print(f"\n🎯 Evaluation Summary:")
        print(f"  📊 Total tasks: {total_tasks}")
        print(f"  ✅ Successful: {success_count}")
        print(f"  📈 Success rate: {success_rate:.2%}")
        print(f"  🔄 Average iterations: {avg_iterations:.1f}")
        
        return summary
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            try:
                self.driver.quit()
                print("✅ WebDriver closed successfully")
            except Exception as e:
                print(f"⚠️ Error closing WebDriver: {e}")


def main():
    """
    Main function for running the test system
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Frontend Design Agent Test System")
    parser.add_argument("--hostname", default="localhost", help="SGLang server hostname")
    parser.add_argument("--code_port", type=int, default=8000, help="Code generation agent port")
    parser.add_argument("--visual_port", type=int, default=8001, help="Visual analysis agent port")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum iterations per task")
    parser.add_argument("--output_path", help="Path to save evaluation results")
    parser.add_argument("--temp_path", help="Temporary directory for files")
    parser.add_argument("--dataset_version", default="v0.2", help="WebSight dataset version")
    
    args = parser.parse_args()
    
    # Create test system
    test_system = FrontendDesignAgentGraph(
        hostname=args.hostname,
        code_port=args.code_port,
        visual_port=args.visual_port,
        max_iterations=args.max_iterations,
        temp_path=args.temp_path
    )
    
    try:
        # Run evaluation
        results = test_system.run_evaluation(
            num_samples=args.num_samples,
            output_path=args.output_path,
            dataset_version=args.dataset_version
        )
        
        if "error" in results:
            print(f"❌ Evaluation failed: {results['error']}")
            return 1
        
        print("🎉 Evaluation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    finally:
        test_system.cleanup()


if __name__ == "__main__":
    exit(main()) 