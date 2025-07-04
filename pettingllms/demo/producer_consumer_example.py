"""
Example demonstrating the producer-consumer mechanism for multi-turn rollouts.

This script shows how to use the LLMAgentProxy with both training and validation modes,
implementing the producer-consumer pattern for efficient multi-turn interactions.

Author: AI Assistant
Date: 2025-01-XX
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import uuid
from typing import Dict, List

try:
    import hydra
    from transformers import AutoTokenizer
    from verl import DataProto
    from pettingllms.llm_agent.agent_proxy import LLMAgentProxy, VllmWrapperWg, ApiCallingWrapperWg
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


class MockDataProto:
    """Mock DataProto for demonstration when verl is not available"""
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch or {}
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}


class DemoLLMAgentProxy:
    """
    Demonstration version of LLMAgentProxy for showing the producer-consumer mechanism.
    This simplified version works without the full ML framework dependencies.
    """
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        print("Demo LLM Agent Proxy initialized")
        
    def _get_default_config(self):
        """Create a default configuration for demo purposes"""
        class DefaultConfig:
            class AgentProxy:
                max_turn = 5
                max_buffer_size = 100
                max_concurrent_requests = 10
                
            agent_proxy = AgentProxy()
            
        return DefaultConfig()
    
    def create_rollout_session(self, mode: str = "validation") -> str:
        """Create a new rollout session"""
        session_id = str(uuid.uuid4())
        print(f"Created {mode} session: {session_id}")
        return session_id
    
    def process_turn_producer(self, session_id: str, env_outputs: List[Dict]) -> MockDataProto:
        """
        Producer step: Generate LM inputs from environment outputs.
        This is where environment observations are converted to prompts for the LLM.
        """
        print(f"[Producer] Session {session_id[:8]}... - Processing {len(env_outputs)} environment outputs")
        
        # Simulate prompt generation from environment state
        prompts = []
        for env_output in env_outputs:
            prompt = f"Environment state: {env_output.get('state', 'unknown')}\nWhat action should I take?"
            prompts.append(prompt)
        
        lm_inputs = MockDataProto(
            batch={'prompts': prompts},
            non_tensor_batch={'env_ids': [env['env_id'] for env in env_outputs]},
            meta_info={'turn': env_outputs[0].get('turn', 0)}
        )
        
        print(f"[Producer] Generated {len(prompts)} prompts")
        return lm_inputs
    
    def process_turn_consumer_step1(self, lm_inputs: MockDataProto) -> MockDataProto:
        """
        Consumer step 1: Generate LLM response from inputs.
        This is where the LLM processes prompts and generates responses.
        """
        prompts = lm_inputs.batch.get('prompts', [])
        print(f"[Consumer Step 1] Processing {len(prompts)} prompts with LLM")
        
        # Simulate LLM generation (in real implementation, this would call the actual LLM)
        responses = []
        for i, prompt in enumerate(prompts):
            # Simulate different responses based on prompt content
            if "move" in prompt.lower():
                response = f"<answer>move forward</answer>"
            elif "action" in prompt.lower():
                response = f"<answer>interact with object</answer>"
            else:
                response = f"<answer>explore area</answer>"
            responses.append(response)
        
        lm_outputs = MockDataProto(
            batch={'responses': responses},
            non_tensor_batch=lm_inputs.non_tensor_batch,
            meta_info=lm_inputs.meta_info
        )
        
        print(f"[Consumer Step 1] Generated {len(responses)} responses")
        return lm_outputs
    
    def process_turn_consumer_step2(self, session_id: str, lm_outputs: MockDataProto) -> List[Dict]:
        """
        Consumer step 2: Generate environment inputs from LLM outputs.
        This is where LLM responses are converted to environment actions.
        """
        responses = lm_outputs.batch.get('responses', [])
        env_ids = lm_outputs.non_tensor_batch.get('env_ids', [])
        
        print(f"[Consumer Step 2] Session {session_id[:8]}... - Converting {len(responses)} responses to actions")
        
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            # Extract action from response (simplified parsing)
            action = "unknown"
            if "<answer>" in response and "</answer>" in response:
                action_content = response.split("<answer>")[1].split("</answer>")[0].strip()
                action = action_content
            
            env_input = {
                'env_id': env_id,
                'action': action,
                'llm_response': response
            }
            env_inputs.append(env_input)
        
        print(f"[Consumer Step 2] Generated {len(env_inputs)} environment actions")
        return env_inputs
    
    def simulate_environment_step(self, env_inputs: List[Dict], turn: int) -> List[Dict]:
        """
        Simulate environment stepping based on actions.
        In real implementation, this would be handled by the environment manager.
        """
        print(f"[Environment] Processing {len(env_inputs)} actions for turn {turn}")
        
        env_outputs = []
        for env_input in env_inputs:
            # Simulate environment response to action
            action = env_input['action']
            env_id = env_input['env_id']
            
            # Simulate different outcomes
            if turn >= 3:  # Some environments finish after 3 turns
                if env_id % 2 == 0:  # Even env_ids finish
                    continue
            
            new_state = f"state_after_{action}_turn_{turn}"
            reward = 1.0 if "forward" in action else 0.5
            
            env_output = {
                'env_id': env_id,
                'state': new_state,
                'reward': reward,
                'turn': turn + 1,
                'done': turn >= 4  # Finish after 5 turns
            }
            env_outputs.append(env_output)
        
        print(f"[Environment] Generated {len(env_outputs)} new states")
        return env_outputs
    
    def rollout_async(self, mode: str = "validation") -> Dict:
        """
        Perform asynchronous multi-turn rollout using producer-consumer mechanism.
        
        Args:
            mode: "train" or "validation"
            
        Returns:
            rollout_results: Dictionary containing rollout information
        """
        print(f"\n=== Starting {mode.upper()} Rollout ===")
        session_id = self.create_rollout_session(mode)
        
        # Initialize environment states
        initial_env_outputs = [
            {'env_id': i, 'state': f'initial_state_{i}', 'turn': 0, 'done': False}
            for i in range(4)  # 4 parallel environments
        ]
        
        env_outputs = initial_env_outputs
        turn_results = []
        
        for turn in range(self.config.agent_proxy.max_turn):
            if len(env_outputs) == 0:  # All environments finished
                print(f"[Turn {turn}] All environments completed")
                break
                
            print(f"\n--- Turn {turn} ---")
            
            # Producer: Generate prompts from environment state
            lm_inputs = self.process_turn_producer(session_id, env_outputs)
            
            # Consumer Step 1: LLM Generation
            lm_outputs = self.process_turn_consumer_step1(lm_inputs)
            
            # Consumer Step 2: Environment Interaction
            env_inputs = self.process_turn_consumer_step2(session_id, lm_outputs)
            
            # Update environment and get new observations
            env_outputs = self.simulate_environment_step(env_inputs, turn)
            
            # Store turn results
            turn_result = {
                'turn': turn,
                'num_active_envs': len(env_inputs),
                'actions_taken': [env_input['action'] for env_input in env_inputs],
                'rewards': [env_output.get('reward', 0) for env_output in env_outputs]
            }
            turn_results.append(turn_result)
        
        # Compute final results
        total_rewards = sum(sum(turn['rewards']) for turn in turn_results)
        total_turns = len(turn_results)
        
        rollout_results = {
            'session_id': session_id,
            'mode': mode,
            'total_turns': total_turns,
            'total_rewards': total_rewards,
            'avg_reward_per_turn': total_rewards / max(total_turns, 1),
            'turn_details': turn_results
        }
        
        print(f"\n=== {mode.upper()} Rollout Complete ===")
        print(f"Total turns: {total_turns}")
        print(f"Total rewards: {total_rewards:.2f}")
        print(f"Average reward per turn: {rollout_results['avg_reward_per_turn']:.2f}")
        
        return rollout_results
    
    def train_step(self) -> Dict:
        """
        Perform a training step with parameter updates.
        In validation mode, this only performs rollouts.
        In train mode, this would also update model parameters.
        """
        print("\n" + "="*50)
        print("TRAINING STEP")
        print("="*50)
        
        # Perform rollout in train mode
        rollout_results = self.rollout_async(mode="train")
        
        # Simulate parameter update (in real implementation, this would update the LLM)
        if rollout_results['mode'] == "train":
            print("\n[Parameter Update] Simulating model parameter updates...")
            print("[Parameter Update] Computing policy gradients from rollout data...")
            print("[Parameter Update] Updating model weights...")
            print("[Parameter Update] Parameter update completed")
        
        training_metrics = {
            "rollout_completed": True,
            "mode": rollout_results['mode'],
            "num_episodes": len(rollout_results['turn_details']),
            "avg_reward": rollout_results['avg_reward_per_turn'],
            "total_reward": rollout_results['total_rewards'],
            "parameter_update_completed": rollout_results['mode'] == "train"
        }
        
        return training_metrics
    
    def validation_step(self) -> Dict:
        """
        Perform a validation step (rollout only, no parameter updates).
        """
        print("\n" + "="*50)
        print("VALIDATION STEP")
        print("="*50)
        
        # Perform rollout in validation mode
        rollout_results = self.rollout_async(mode="validation")
        
        validation_metrics = {
            "rollout_completed": True,
            "mode": rollout_results['mode'],
            "num_episodes": len(rollout_results['turn_details']),
            "avg_reward": rollout_results['avg_reward_per_turn'],
            "total_reward": rollout_results['total_rewards'],
            "parameter_update_completed": False  # No updates in validation
        }
        
        return validation_metrics


def demonstrate_producer_consumer_mechanism():
    """
    Main demonstration function showing the producer-consumer mechanism.
    """
    print("Producer-Consumer Multi-Turn Rollout Demonstration")
    print("=" * 60)
    print()
    print("This demo shows how the LLMAgentProxy implements:")
    print("1. Producer: Converts environment states to LLM prompts")
    print("2. Consumer Step 1: Processes prompts with LLM to generate responses")
    print("3. Consumer Step 2: Converts LLM responses to environment actions")
    print("4. Environment updates based on actions")
    print()
    print("Two modes are supported:")
    print("- Validation: Only performs rollouts for evaluation")
    print("- Train: Performs rollouts AND parameter updates")
    print()
    
    # Initialize demo agent
    agent = DemoLLMAgentProxy()
    
    # Demonstrate validation mode
    validation_metrics = agent.validation_step()
    
    # Demonstrate training mode
    training_metrics = agent.train_step()
    
    # Compare results
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"Validation - Avg Reward: {validation_metrics['avg_reward']:.2f}, "
          f"Episodes: {validation_metrics['num_episodes']}, "
          f"Param Updates: {validation_metrics['parameter_update_completed']}")
    print(f"Training   - Avg Reward: {training_metrics['avg_reward']:.2f}, "
          f"Episodes: {training_metrics['num_episodes']}, "
          f"Param Updates: {training_metrics['parameter_update_completed']}")
    print()
    print("Key Differences:")
    print("- Validation mode: Only evaluates current policy performance")
    print("- Training mode: Evaluates AND improves policy through parameter updates")
    print("- Both modes use the same producer-consumer pipeline for efficiency")


if __name__ == "__main__":
    demonstrate_producer_consumer_mechanism() 