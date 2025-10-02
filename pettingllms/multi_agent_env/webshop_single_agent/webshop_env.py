import logging
import copy
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass

from pettingllms.multi_agent_env.base.env import Env
from webshop_minimal import WebAgentTextEnv, init_basedir
import random
import string
import uuid

logger = logging.getLogger(__name__)

@dataclass
class WebShopEnvState:
    """State class for WebShop environment"""
    goal: str = None  # Shopping goal/instruction
    current_observation: str = None  # Current page observation
    available_actions: List[str] = None  # Available actions at current step
    action_history: List[str] = None  # History of actions taken
    observation_history: List[str] = None  # History of observations
    reward_history: List[float] = None  # History of rewards
    current_reward: float = 0.0  # Current step reward
    success: bool = False  # Whether task was completed successfully
    purchase_made: bool = False  # Whether a purchase was made
    target_product: str = None  # Target product information
    session_id: str = None  # WebShop session ID
    step_count: int = 0  # Number of steps taken
    max_steps: int = 10  # Maximum allowed steps

# Define global constant for render instructions
RENDER_INSTRUCTIONS = [
    "We must buy a product within 10 actions. It doesn't have to match perfectly with description.",
    "Search term should not include details like size, color.",
    "Never search for more than 2 times.",
    "Do not be too strict about the description, it's more important to buy one that is close enough within action limit.",
    "Prioritize click a product in the current page over going to next page.",
    "Almost never click[next >] for more than 2 times."
    "Almost never click[< prev] unless you are sure the product is on one of the previous pages.",
    "If you have less than 3 actions left, just buy the first product you see in the current page.",
    "If an matching option exists, make sure to click[size] then click[color], one at a time, before click[buy now], but don't have to if only 1 action left, in that case you just click[buy now]. Never click description."
]


class WebShopEnv(Env):
    """
    WebShop environment for single agent shopping tasks.
    
    This environment coordinates shopping interactions where an agent needs to
    navigate a webshop to find and purchase products based on given goals.
    """
    
    def __init__(
        self, 
        env_idx: int,
        rollout_idx: int,
        max_turns: int,
        config: dict | None = None,
    ):
        """
        Initialize the WebShop environment.
        
        Args:
            env_idx: Environment index
            rollout_idx: Rollout index  
            max_turns: Maximum number of turns before terminating
            config: Configuration for the environment
        """
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, max_turns=max_turns, config=config)
        self.state = WebShopEnvState()
        
        # Initialize WebShop-specific configuration
        webshop_config = config.get('webshop', {}) if config else {}
        self.observation_mode = webshop_config.get('observation_mode', 'text')
        self.file_path = webshop_config.get('file_path', None)
        self.server = webshop_config.get('server', None)
        self.filter_goals = webshop_config.get('filter_goals', True)
        self.limit_goals = webshop_config.get('limit_goals', -1)
        self.num_products = webshop_config.get('num_products', 100)
        self.human_goals = webshop_config.get('human_goals', False)
        self.show_attrs = webshop_config.get('show_attrs', True)
        self.render_cache = None
        
        # Initialize dataset if provided
        dataset = webshop_config.get('dataset', None)
        if dataset:
            init_basedir(dataset)

        # Initialize WebAgentTextEnv
        self.webshop_env = WebAgentTextEnv(
            observation_mode=self.observation_mode,
            file_path=self.file_path,
            server=self.server,
            filter_goals=self.filter_goals,
            limit_goals=self.limit_goals,
            num_products=self.num_products,
            human_goals=self.human_goals,
            show_attrs=self.show_attrs,
            session_prefix=str(uuid.uuid4().hex), # we use a random session prefix to avoid collision
        )

    def _get_permuted_index(self, idx, seed=42):
        """Map index to a deterministically permuted index in the same range.
        
        Args:
            idx: The original index
            seed: Random seed to ensure deterministic permutation
            
        Returns:
            int: The permuted index
        """
        # Create a cache key based on goals length and seed
        cache_key = f"perm_{len(self.server.goals)}_{seed}"
        
        # Create or retrieve the permutation map
        if not hasattr(self, cache_key):
            # Initialize with fixed seed
            rng = random.Random(seed)
            
            # Generate the full permutation
            indices = list(range(len(self.server.goals)))
            rng.shuffle(indices)
            
            # Store the permutation as an instance attribute
            setattr(self, cache_key, indices)
        
        # Look up the permuted index
        permutation = getattr(self, cache_key)
        return permutation[idx]

    def reset(self, seed=None, mode="train", session: Optional[Union[str, int]] = None, instruction_text: Optional[str] = None):
        """
        Reset the environment and return the initial observation.

        Args:
            seed: Random seed for reproducibility
            mode: Environment mode ("train", "val", "test")
            session: The new session ID
            instruction_text: Optional new instruction text

        Returns:
            The environment state after reset
        """
        # Reset parent class
        super().reset()
        
        # Reset state
        self.state = WebShopEnvState()
        self.state.action_history = []
        self.state.observation_history = []
        self.state.reward_history = []
        self.state.max_steps = self.max_turns
        
        if seed is None:
            return self
            
        # Determine goal index based on mode
        if mode == "test":
            goal_idx = seed % 500
        elif mode == "val":
            goal_idx = seed % 1000 + 500
        elif mode == "train":
            goal_idx = seed % (len(self.webshop_env.server.goals) - 1500) + 1500
            
        session = self._get_permuted_index(goal_idx) if session is None else session
        self.state.session_id = str(session)
        
        # Reset WebShop environment
        obs, _ = self.webshop_env.reset(session=session, instruction_text=instruction_text)
        
        # Update state with initial observation
        self.state.current_observation = obs
        self.state.goal = self.webshop_env.get_instruction_text()
        self.state.available_actions = self.get_available_actions()
        self.state.observation_history.append(obs)
        
        self.prepare_render_cache(self.state.goal)
        return self

    async def step(self, role: str, action: str, env_worker: Any = None):
        """
        Execute an action in the WebShop environment.
        
        Args:
            role: Agent role (should be "webshop_agent")
            action: Action to take in the environment
            env_worker: Optional environment worker for parallel execution
        """
        if role == "webshop_agent":
            await self._webshop_step(action, env_worker)
        else:
            raise ValueError(f"Invalid role: {role}")

    async def _webshop_step(self, action: str, env_worker: Any = None):
        """
        Execute a WebShop action and update the environment state.
        
        Args:
            action: Action string to execute
            env_worker: Optional environment worker
        """
        # Store previous state for comparison
        last_observation = self.state.current_observation
        
        # Check if action is valid
        action_is_valid = (action in self.state.available_actions or 
                          ("search[<content>]" in self.state.available_actions and 
                           action.startswith('search[') and action.endswith(']')))
        
        # Execute action in WebShop environment
        obs, reward, done, info = self.webshop_env.step(action)
        
        # Update state
        self.state.step_count += 1
        self.state.current_observation = obs
        self.state.current_reward = reward
        self.state.action_history.append(action)
        self.state.observation_history.append(obs)
        self.state.reward_history.append(reward)
        self.state.available_actions = self.get_available_actions()
        
        # Update success flags
        self.state.success = reward == 1
        self.state.purchase_made = done
        
        # Check termination conditions
        if done or self.state.step_count >= self.state.max_steps:
            self.done = True
            self.is_pass = self.state.success
            if done:
                self.termination_reason = "purchase_completed"
            else:
                self.termination_reason = "max_steps_reached"
        
        # Prepare render cache
        self.prepare_render_cache(obs)
        
        # Update additional info
        info = (info or {}).copy()
        info.update({
            "reward": reward,
            "action_is_effective": obs != last_observation,
            "action_is_valid": action_is_valid,
            "success": 1 if reward == 1 else 0,
            "success_purchase": 1 if done else 0,
            "success_find": 1 if reward == 1 else 0,
            "end_of_page": 1 if tuple(self.state.available_actions) == ('click[back to search]', 'click[< prev]') else 0,
            "step_count": self.state.step_count,
        })
        
        logger.info(f"WebShop step {self.state.step_count}: action='{action}', reward={reward}, done={done}")
        
        return info

    def render(self, mode=None):
        """
        Render the environment.
        """
        return self.render_cache

    def close(self):
        """
        Close the environment.
        """
        self.webshop_env.close()

    def prepare_render_cache(self, observation: str):
        """
        Prepare the render cache for the environment.
        """
        available_actions = self.get_available_actions()
        self.render_cache = observation + "."
        self.render_cache += "\n".join(RENDER_INSTRUCTIONS)
        self.render_cache += "\n You must choose from these actions:" + ", ".join(available_actions) + "."
        

    def get_available_actions(self):
        """
        Parse the available actions in the environment to a list of strings.
        """
        orig_available_actions = self.webshop_env.get_available_actions()
        available_actions = []

        if orig_available_actions['has_search_bar']:
            available_actions.append('search[<content>]')

        for clickable in orig_available_actions['clickables']:
            if clickable != 'search':
                available_actions.append(f'click[{clickable}]')
        # TODO: we may need to purge the case when available_actions == ['click[back to search]', 'click[< prev]', 'click[next >]']
        is_end_of_page = tuple(available_actions) == ('click[back to search]', 'click[< prev]', 'click[next >]')
        if is_end_of_page:
            available_actions.remove('click[next >]')
        return available_actions


class WebShopEnvBatch:
    """Batch environment manager for WebShop environments."""
    
    def __init__(self, env_idx_list: List[int], rollout_idx_list: List[int], samples: int, max_turns: int, config: dict, mode="train", *, env_workers: List = None):
        """
        Initialize batch of WebShop environments.
        
        Args:
            env_idx_list: List of environment indices
            rollout_idx_list: List of rollout indices
            samples: Number of samples per environment
            max_turns: Maximum turns per environment
            config: Configuration dictionary
            mode: Environment mode ("train", "val", "test")
            env_workers: Optional list of environment workers
        """
        self.env_idx_list = env_idx_list
        self.rollout_idx_list = rollout_idx_list
        self.samples = samples
        self.max_turns = max_turns
        self.config = config
        self.mode = mode
        self.env_workers = env_workers or []
        
        # Create environments
        self.envs = []
        for i, (env_idx, rollout_idx) in enumerate(zip(env_idx_list, rollout_idx_list)):
            env = WebShopEnv(
                env_idx=env_idx,
                rollout_idx=rollout_idx,
                max_turns=max_turns,
                config=config
            )
            self.envs.append(env)
    
    def reset_all(self, seeds: List[int] = None):
        """Reset all environments in the batch."""
        if seeds is None:
            seeds = list(range(len(self.envs)))
        
        results = []
        for i, (env, seed) in enumerate(zip(self.envs, seeds)):
            result = env.reset(seed=seed, mode=self.mode)
            results.append(result)
        
        return results
    
    async def step_all(self, actions: List[tuple]):
        """
        Execute actions in all environments.
        
        Args:
            actions: List of (role, action) tuples for each environment
        """
        results = []
        for env, (role, action) in zip(self.envs, actions):
            env_worker = self.env_workers[len(results)] if len(results) < len(self.env_workers) else None
            result = await env.step(role, action, env_worker)
            results.append(result)
        
        return results
    
    def close_all(self):
        """Close all environments in the batch."""
        for env in self.envs:
            env.close()


if __name__ == '__main__':
    config = {
        'webshop': {
            'observation_mode': 'text',
            'filter_goals': True,
            'limit_goals': -1,
            'num_products': 100,
            'human_goals': False,
            'show_attrs': True,
        }
    }
    
    env = WebShopEnv(env_idx=0, rollout_idx=0, max_turns=10, config=config)
    env.reset(seed=42, mode="train")
    print(f"Goal: {env.state.goal}")
    print(f"Initial observation: {env.state.current_observation}")
    
    import asyncio
    
    async def test_env():
        while not env.done:
            print(f"\nStep {env.state.step_count}")
            print(f"Observation: {env.state.current_observation}")
            print(f"Available actions: {env.state.available_actions}")
            action = input("Enter action (or 'q' to quit): ")
            if action == 'q':
                break
            
            try:
                info = await env.step("webshop_agent", action)
                print(f"Reward: {env.state.current_reward}")
                print(f"Done: {env.done}")
                print(f"Info: {info}")
            except Exception as e:
                print(f"Error: {e}")
                
    asyncio.run(test_env())
    env.close()
