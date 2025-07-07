import matplotlib.pyplot as plt
import numpy as np
import os
import pyspiel
import gym
import re
import requests
import json
import logging
from datetime import datetime

# Optional import of sglang
try:
    from sglang.engine import Engine
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

# from pettingllms.env.base import BaseDiscreteActionEnv
from pettingllms.env.tic_tac_toe.config import TicTacToeEnvConfig
from pettingllms.env.tic_tac_toe.prompt import  extract_action, Observations, get_prompt_for_self_play,Render

# Setup logging
def setup_logging():
    """Setup two loggers: one for game results, one for LLM conversations"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Game results logger
    game_logger = logging.getLogger('game_results')
    game_logger.setLevel(logging.INFO)
    game_handler = logging.FileHandler(f"logs/game_results_{timestamp}.log")
    game_formatter = logging.Formatter('%(asctime)s - %(message)s')
    game_handler.setFormatter(game_formatter)
    game_logger.addHandler(game_handler)
    
    # LLM conversation logger
    llm_logger = logging.getLogger('llm_conversation')
    llm_logger.setLevel(logging.INFO)
    llm_handler = logging.FileHandler(f"logs/llm_conversation_{timestamp}.log")
    llm_formatter = logging.Formatter('%(asctime)s - %(message)s')
    llm_handler.setFormatter(llm_formatter)
    llm_logger.addHandler(llm_handler)
    
    return game_logger, llm_logger

# Initialize loggers
game_logger, llm_logger = setup_logging()

class TicTacToe:
    """Tic-Tac-Toe game environment using OpenSpiel."""

    def __init__(self, config=None, **kwargs):
        """ 
        config attributes:
            visual_obs: bool = True
            image_dir: Optional[str] = None
            recording_type: str = 'gif'
            recording_fps: int = 2
            render_mode: str = "text"
            llm_play_mode: str="self_play"
            action_lookup: Optional[Dict[int, str]] = field(default_factory=lambda: {
                0: "X(0,0)", 1: "X(0,1)", 2: "X(0,2)",
                3: "X(1,0)", 4: "X(1,1)", 5: "X(1,2)", 
                6: "X(2,0)", 7: "X(2,1)", 8: "X(2,2)"
            })
            grid_lookup: Optional[Dict[str, str]] = field(default_factory=lambda: {
                ".": "empty", "x": "X", "o": "O"
            }) 
        """
        self.config = config or TicTacToeEnvConfig()
        self.visual_obs = self.config.visual_obs
        self.image_dir = self.config.image_dir
        self.recording_type = self.config.recording_type
        self.recording_fps = self.config.recording_fps
        self.render_mode = self.config.render_mode
        self.llm_play_mode = self.config.llm_play_mode
        self.action_lookup = self.config.action_lookup
        self.grid_lookup = self.config.grid_lookup
       
        
        
        
        self._env = pyspiel.load_game("tic_tac_toe")
        self.state = self._env.new_initial_state()
        self.num_agents = self._env.num_players()
        self.observations = Observations(
            is_terminal=self.state.is_terminal(),
            agent_id=self.current_player,
            reward={0:0,1:0},
            legal_actions=self.state.legal_actions(self.current_player),
            action_lookup=self.action_lookup,
            state_render=Render(text=str(self.state),image_paths=[])
        )
        self.image_paths = self._save_image() if self.visual_obs else []
        self.initial_prompt = get_prompt_for_self_play(self.observations)
        

    @property
    def current_player(self):
        """Get the current player."""
        return self.state.current_player()

    def reset(self, seed=None, mode=None):
        """Reset the environment and return the initial observation."""
        self.state = self._env.new_initial_state()
        self.observations = Observations(
            is_terminal=self.state.is_terminal(),
            agent_id=self.current_player,
            reward={0:0,1:0},
            legal_actions=self.state.legal_actions(self.current_player),
            action_lookup=self.action_lookup,
            state_render=Render(text=str(self.state),image_paths=[])
        )
        
    
    def play_step(self, agent_action: str)->str:
        action = extract_action(self.current_player, agent_action,self.observations.legal_actions)
        self._env_step(action)
        agent_prompt = get_prompt_for_self_play(self.observations)
        
        return agent_prompt.text

    
    def agent_step(self, agent_action: str)->str:
        if self.llm_play_mode == "self_play":
            return self.play_step(agent_action)
        elif self.llm_play_mode == "competition_play":
            return self.play_step(agent_action)
    
        else:
            raise ValueError(f"Invalid llm_play_mode: {self.llm_play_mode}")


    def _env_step(self, action: int):
        """Execute one step in the environment."""
       
        current_player_id = self.current_player
        
        # Check if action is valid
        legal_actions = self.state.legal_actions(current_player_id)
        if action not in legal_actions:
            return self.render(), 0, False, {"action_is_effective": False, "action_is_valid": False, "success": False}
        
        # Apply action
        self.state.apply_action(action)
        
        
        
        self.observations.agent_id=self.state.current_player()
        self.observations.legal_actions=self.state.legal_actions(self.current_player)
        self.observations.is_terminal=self.state.is_terminal()
        self.observations.reward={0:0,1:0}
        if self.state.is_terminal():
            self.observations.reward=self.state.returns()
        self.observations.state_render.text = str(self.state)

        
        self.observations.state_render.image_paths = self._save_image()
        


    def save_txt(self, mode=None)->str:
                    # Convert pyspiel state to text representation
        board_str = str(self.state).strip()
        lines = board_str.split('\n')
        
        # Create a cleaner text representation
        result = []
        for line in lines:
            clean_line = ""
            for char in line:
                if char == '.':
                    clean_line += "_"
                elif char in ['x', 'o']:
                    clean_line += char.upper()
                else:
                    clean_line += char
            result.append(clean_line)
        
    def render(self, mode=None):
        """Render the current state of the game."""
        if self.render_mode == "text" or mode == "text":
            return str(self.state)
        else:
            return str(self.state)

    def _get_winner(self):
        """Get the winner of the game."""
        if not self.state.is_terminal():
            return None
        returns = self.state.returns()
        if returns[0] > returns[1]:
            return 0
        elif returns[1] > returns[0]:
            return 1
        else:
            return -1  # Draw


    def _save_image(self):
        """Save current state as image."""
        if not self.visual_obs:
            return []
            
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.invert_yaxis()
        
        # Draw grid lines
        for x in range(1, 3):
            ax.plot([x - 0.5, x - 0.5], [-0.5, 2.5], color='black', linewidth=2)
        for y in range(1, 3):
            ax.plot([-0.5, 2.5], [y - 0.5, y - 0.5], color='black', linewidth=2)
        
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['0', '1', '2'])
        ax.set_yticklabels(['0', '1', '2'])
        
        # Draw pieces
        board = np.array([list(line) for line in str(self.state).strip().split("\n")])
        for i in range(3):
            for j in range(3):
                piece = board[i][j]
                if piece != '.':
                    color = 'red' if piece == 'x' else 'blue'
                    ax.text(j, i, piece.upper(), fontsize=30, ha='center', va='center', color=color)
        
        ax.set_aspect('equal')

        # Ensure image_dir exists
        if self.image_dir and not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir, exist_ok=True)
        
        image_path = os.path.join(self.image_dir or ".", f"step_{self.state.move_number()}.png")
        plt.savefig(image_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Only add frame when recorders exist
        if hasattr(self, 'recorders') and self.recorders:
            self.recorders[0].add_frame(image_path)
        
        return [image_path]

    def close(self):
        """Close the environment."""
        if hasattr(self, 'recorders'):
            for recorder in self.recorders:
                recorder.clear()


if __name__ == '__main__':
    
    def generate_tic_tac_toe_move(prompt: str) -> dict:
        """
        Send request to local sglang service to generate tic-tac-toe move
        
        Args:
            prompt: Game prompt text
            
        Returns:
            dict: Dictionary containing response
        """
        url = "http://localhost:30000/generate"
        
        # Use SGLang /generate endpoint format
        data = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 128,
                "stop": ["\n\n", "Human:", "Assistant:"]
            }
        }
        
        try:
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Handle SGLang /generate endpoint response format
            if "text" in result:
                response_text = result["text"]
                llm_logger.info(f"LLM Response: {response_text}")
                return {"response": response_text}
            elif "generated_text" in result:
                response_text = result["generated_text"]
                llm_logger.info(f"LLM Response: {response_text}")
                return {"response": response_text}
            elif "choices" in result and len(result["choices"]) > 0:
                # Sometimes it may also return choices format
                choice = result["choices"][0]
                if "text" in choice:
                    response_text = choice["text"]
                    llm_logger.info(f"LLM Response: {response_text}")
                    return {"response": response_text}
            
            response_text = str(result)
            llm_logger.info(f"LLM Response (fallback): {response_text}")
            return {"response": response_text}
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Request to sglang service failed: {e}"
            llm_logger.error(error_msg)
            return {"response": "Error: Unable to connect to sglang service"}
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse response: {e}"
            llm_logger.error(error_msg)
            return {"response": "Error: Response format error"}
        except Exception as e:
            error_msg = f"Other error: {e}"
            llm_logger.error(error_msg)
            return {"response": f"Error: {str(e)}"}
    
    def test_self_play(log):
        """Test self-play mode"""
        game_logger.info("=== Starting self-play mode ===")
        config = TicTacToeEnvConfig(visual_obs=False, render_mode='text', llm_play_mode="self_play")
        env = TicTacToe(config)
        
        game_logger.info("Game initialized")
        current_state = env.reset()
        
        # Initialize prompt
        current_prompt = env.initial_prompt.text
        llm_logger.info(f"Initial prompt: {current_prompt}")
       
        max_steps = 9  # Tic-tac-toe has at most 9 steps
        for step in range(max_steps):
            if env.state.is_terminal():
                break
                
            game_logger.info(f"Step {step + 1}: Player {env.current_player}")
            llm_logger.info(f"Step {step + 1} prompt (length: {len(current_prompt)}): {current_prompt}")
            
            # Use LLM to generate action
            try:
                response = generate_tic_tac_toe_move(current_prompt)
            except Exception as e:
                llm_logger.error(f"LLM generation failed: {e}")
                response = {"response": ""}
                
            agent_action = response["response"] if response else ""
            llm_logger.info(f"Player {env.current_player} action: {agent_action}")
            
            # Execute action and get next prompt
            next_prompt = env.agent_step(agent_action)
            current_prompt = next_prompt
            
            game_state = env.render()
            game_logger.info(f"Game state after step {step + 1}:\n{game_state}")
                
        # Display final result
        if env.state.is_terminal():
            winner = env._get_winner()
            if winner == -1:
                result = "Game ended: Draw"
            else:
                result = f"Game ended: Player {winner} wins"
            game_logger.info(result)
            return result
        else:
            result = "Game ended: Incomplete"
            game_logger.info(result)
            return result
    
    def test_competition_play():
        """Test competition play mode"""
        game_logger.info("=== Starting competition play mode ===")
        config = TicTacToeEnvConfig(visual_obs=False, render_mode='text', llm_play_mode="competition_play")
        env = TicTacToe(config)
        
        game_logger.info("Game initialized")
        current_state = env.reset()
        
        current_prompt = env.initial_prompt.text
        llm_logger.info(f"Initial prompt: {current_prompt}")
        
        max_steps = 9  # Tic-tac-toe has at most 9 steps
        for step in range(max_steps):
            if env.state.is_terminal():
                break
                
            game_logger.info(f"Step {step + 1}: Player {env.current_player}")
            llm_logger.info(f"Step {step + 1} prompt (length: {len(current_prompt)})")
            
            # Use LLM to generate action
            try:
                response = generate_tic_tac_toe_move(current_prompt)
                agent_action = response["response"]
                llm_logger.info(f"Player {env.current_player} action: {agent_action}")
                
                # Execute action and get next prompt
                next_prompt = env.agent_step(agent_action)
                current_prompt = next_prompt
                
                game_state = env.render()
                game_logger.info(f"Game state after step {step + 1}:\n{game_state}")
                
            except Exception as e:
                llm_logger.error(f"Error in step {step + 1}: {e}")
                break
        
        # Display final result
        if env.state.is_terminal():
            winner = env._get_winner()
            if winner == -1:
                result = "Game ended: Draw"
            else:
                result = f"Game ended: Player {winner} wins"
            game_logger.info(result)
            return result
        else:
            result = "Game ended: Incomplete"
            game_logger.info(result)
            return result
    
  
    # Run tests
    game_logger.info("Starting tic-tac-toe environment test...")
    
    # Check if sglang service is available
    try:
        test_response = requests.get("http://localhost:30000/health", timeout=5)
        if test_response.status_code == 200:
            game_logger.info("Sglang service connection successful!")
            sglang_available = True
        else:
            game_logger.warning(f"Sglang service unavailable, status code: {test_response.status_code}")
            sglang_available = False
    except Exception as e:
        game_logger.warning(f"Unable to connect to sglang service: {e}")
        game_logger.info("Trying to connect to http://localhost:30000/generate for testing...")
        sglang_available = True  # Even if health check fails, try game testing
    
    if sglang_available:
        game_logger.info("Using local sglang service (port 30000)...")
        
        # Create summary file for game results
        summary_file = "logs/game_summary_Qwen2.5-1.5B-Instruct.txt"
        
        for i in range(10):
            game_logger.info(f"Starting Round {i+1}")
            result = test_self_play(None)
            
            # Write summary to file
            with open(summary_file, "a") as f:
                f.write(f"Round {i+1}: {result}\n")
                
        game_logger.info("All rounds completed")
    else:
        game_logger.error("Sglang service unavailable, skipping LLM play test")
  
