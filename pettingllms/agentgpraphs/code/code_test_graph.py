import logging
from typing import Any, Dict, Optional, Tuple

from pettingllms.agent_graphs.code.code_agent import CodeGenerationAgent
from pettingllms.agent_graphs.code.unit_test_agent import UnitTestGenerationAgent
from pettingllms.environments.code.competition_coding import CompetitionCodingEnv

logger = logging.getLogger(__name__)


class CodeTestAgentGraph:
    """
    Graph structure coordinating code generation agent and unit test generation agent 
    interactions with the environment.
    
    Workflow:
    1. Initially, code generation agent generates code
    2. Then unit test generation agent generates test cases
    3. Each agent generation is followed by calling environment's step method
    4. Alternates until done or maximum steps reached
    """

    def __init__(
        self, 
        max_steps: int = 10,
        code_agent_config: Optional[Dict] = None,
        test_agent_config: Optional[Dict] = None,
        env_config: Optional[Dict] = None,
        validate_mode: str = "using reference"
    ):
        """
        Initialize the code test agent graph.

        Args:
            max_steps: Maximum step limit
            code_agent_config: Configuration for code generation agent
            test_agent_config: Configuration for test generation agent  
            env_config: Environment configuration
            validate_mode: Validation mode ("using reference" or others)
        """
        self.max_steps = max_steps
        self.validate_mode = validate_mode
        self.current_step = 0
        self.done = False
        
        # Initialize agents
        code_config = code_agent_config or {}
        test_config = test_agent_config or {}
        env_conf = env_config or {}
        
        self.code_agent = CodeGenerationAgent(**code_config)
        self.test_agent = UnitTestGenerationAgent(**test_config)
        self.env = CompetitionCodingEnv(**env_conf)
        
        # Track state
        self.current_agent = "code"  # "code" or "test"
        self.trajectory_history = []
        self.total_reward = 0.0
        
        logger.info(f"Initialized code test agent graph with max steps: {max_steps}")

    def reset(self, task: Dict) -> Dict:
        """
        Reset graph state and start a new task.

        Args:
            task: Task dictionary containing problem description and other info

        Returns:
            Initial observation
        """
        self.current_step = 0
        self.done = False
        self.current_agent = "code"
        self.trajectory_history = []
        self.total_reward = 0.0
        
        # Reset agents and environment
        self.code_agent.reset()
        self.test_agent.reset()
        initial_obs, _ = self.env.reset(task=task)
        
        logger.info(f"Reset complete, starting task: {task.get('question', 'Unknown')[:50]}...")
        return initial_obs

    def step(self, model_response: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step: model response -> agent processing -> environment step.

        Args:
            model_response: Response from the language model

        Returns:
            Tuple of (next_observation, reward, done, info)
        """
        if self.done:
            logger.warning("Graph already completed, cannot continue step")
            return {}, 0.0, True, {"error": "Graph already done"}

        try:
            # 1. Process model response based on current agent
            if self.current_agent == "code":
                action = self.code_agent.update_from_model(model_response)
                role = "code generation"
                agent_name = "code generation agent"
            else:  # test
                action = self.test_agent.update_from_model(model_response)
                role = "test generation"
                agent_name = "test generation agent"

            logger.info(f"Step {self.current_step + 1}: {agent_name} generated response")

            # 2. Call environment's step method
            next_obs, reward, env_done, info = self.env.step(
                role=role, 
                validate_mode=self.validate_mode, 
                action=action.action
            )

            # 3. Update agents' state
            if self.current_agent == "code":
                self.code_agent.update_from_env(next_obs, reward, env_done, info)
            else:
                self.test_agent.update_from_env(next_obs, reward, env_done, info)

            # 4. Record trajectory
            self.trajectory_history.append({
                "step": self.current_step + 1,
                "agent": self.current_agent,
                "role": role,
                "action": action.action,
                "observation": next_obs,
                "reward": reward,
                "done": env_done,
                "info": info
            })

            self.total_reward += reward
            self.current_step += 1

            # 5. Check termination conditions
            if env_done or self.current_step >= self.max_steps:
                self.done = True
                logger.info(f"Graph execution complete. Reason: {'environment done' if env_done else 'max steps reached'}")
            else:
                # 6. Switch to next agent
                self._switch_agent()

            # 7. If not done, prepare observation for next agent
            if not self.done:
                next_agent_obs = self._prepare_next_agent_observation(next_obs)
                return next_agent_obs, reward, self.done, info
            else:
                return next_obs, reward, self.done, info

        except Exception as e:
            logger.error(f"Step execution error: {str(e)}")
            self.done = True
            return {"error": str(e)}, 0.0, True, {"error": str(e)}

    def _switch_agent(self):
        """Switch current active agent"""
        if self.current_agent == "code":
            self.current_agent = "test"
            logger.debug("Switched to test generation agent")
        else:
            self.current_agent = "code"
            logger.debug("Switched to code generation agent")

    def _prepare_next_agent_observation(self, env_obs: Dict) -> Dict:
        """Prepare observation information for next agent"""
        # Add current environment state to observation
        if "state" not in env_obs:
            env_obs["state"] = self.env.get_state()
        
        # Add question information for test agent if needed
        if self.current_agent == "test" and "question" not in env_obs:
            # Get original question from environment
            if hasattr(self.env, 'task') and self.env.task:
                env_obs["question"] = self.env.task.get("question", "")
        
        return env_obs

    def get_current_agent_trajectory(self) -> Dict:
        """Get current agent's trajectory"""
        if self.current_agent == "code":
            return self.code_agent.trajectory
        else:
            return self.test_agent.trajectory

    def get_all_trajectories(self) -> Dict:
        """Get all agents' trajectories"""
        return {
            "code_agent": self.code_agent.trajectory,
            "test_agent": self.test_agent.trajectory,
            "graph_history": self.trajectory_history,
            "total_reward": self.total_reward,
            "total_steps": self.current_step
        }

    def get_current_state(self) -> Dict:
        """Get current state of the graph"""
        return {
            "current_step": self.current_step,
            "current_agent": self.current_agent,
            "done": self.done,
            "total_reward": self.total_reward,
            "env_state": self.env.get_state(),
            "code_agent_state": self.code_agent.get_current_state(),
            "test_agent_state": self.test_agent.get_current_state()
        }

    def get_next_agent_chat_completions(self) -> list[dict[str, str]]:
        """Get chat history of the next agent that should respond"""
        if self.current_agent == "code":
            return self.code_agent.chat_completions
        else:
            return self.test_agent.chat_completions

    @property
    def is_done(self) -> bool:
        """Check if graph is completed"""
        return self.done

    @property
    def current_agent_name(self) -> str:
        """Get name of current active agent"""
        return "code generation agent" if self.current_agent == "code" else "test generation agent" 