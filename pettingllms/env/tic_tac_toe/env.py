import matplotlib.pyplot as plt
import numpy as np
import os
import pyspiel
import gym
import re

from pettingllms.env.base import BaseDiscreteActionEnv
from pettingllms.env.tic_tac_toe.config import TicTacToeEnvConfig


class TicTacToe(BaseDiscreteActionEnv):
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
        self.state = None
        self.num_agents = self._env.num_players()
        self.image_paths = []
        
        
        
        if self.visual_obs:
            assert self.image_dir is not None, "image_dir must not be None for visual observations."
            from pettingllms.utils.recorder import Recorder
            self.recorders = [Recorder(self.image_dir, self.recording_type, self.recording_fps)]

    @property
    def current_player(self):
        return self.state.current_player()

    def reset(self, seed=None, mode=None):
        """Reset the environment and return the initial observation."""
        self.state = self._env.new_initial_state()
        if self.visual_obs:
            self.recorders[0].clear()
            self.image_paths = self._save_image()
        return self.render()
    
    def self_play_step(self, agent_action: str)->str:
        """

        self_play_step:
            agent_action: str
            return: str
        """
        return agent_action

    
    def agent_step(self, agent_action: str)->str:
        if self.llm_play_mode == "self_play":
            return self.self_play_step(agent_action)
        else:
            raise ValueError(f"Invalid llm_play_mode: {self.llm_play_mode}")


    def _env_step(self, action: int):
        """Execute one step in the environment."""
        if self.state.is_terminal():
            return self.render(), 0, True, {"action_is_effective": False, "action_is_valid": False, "success": False}
        
        # Check if action is valid
        legal_actions = self.state.legal_actions(self.current_player)
        if action not in legal_actions:
            return self.render(), 0, False, {"action_is_effective": False, "action_is_valid": False, "success": False}
        
        # Apply action
        self.state.apply_action(action)
        
        if self.visual_obs:
            self.image_paths = self._save_image()
            if self.state.is_terminal():
                self.recorders[0].save()
        
        # Get reward and check if done
        reward = 0
        done = self.state.is_terminal()
        success = False
        
        if done:
            returns = self.state.returns()
            # Reward: +1 for win, -1 for loss, 0 for draw
            reward = returns[self.current_player] if len(returns) > self.current_player else 0
            success = reward > 0
        
        next_obs = self.render()
        info = {
            "action_is_effective": True, 
            "action_is_valid": True, 
            "success": success,
            "winner": self._get_winner() if done else None
        }
        
        return next_obs, reward, done, info

    def render(self, mode=None):
        """Render the environment."""
        render_mode = mode if mode is not None else self.render_mode
        
        if render_mode == 'text':
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
            
            return '\n'.join(result)
            
        elif render_mode == 'rgb_array':
            if self.visual_obs and self.image_paths:
                # Return the last saved image as numpy array
                import matplotlib.image as mpimg
                return mpimg.imread(self.image_paths[-1])
            else:
                # Generate image on the fly
                return self._generate_rgb_array()
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

    def extract_actions(self, answer: str):
        """Extract valid actions from answer string."""
        # Return all possible actions (0-8 for tic-tac-toe positions)
        return list(range(9))

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

    def _generate_rgb_array(self):
        """Generate RGB array representation of the board."""
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
        
        # Convert to RGB array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return buf

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

        image_path = os.path.join(self.image_dir, f"step_{self.state.move_number()}.png")
        plt.savefig(image_path, dpi=100, bbox_inches='tight')
        plt.close()
        self.recorders[0].add_frame(image_path)
        
        return [image_path]

    def close(self):
        """Close the environment."""
        if hasattr(self, 'recorders'):
            for recorder in self.recorders:
                recorder.clear()
        super().close()


if __name__ == '__main__':
    config = TicTacToeEnvConfig(visual_obs=False, render_mode='text')
    env = TicTacToe(config)
    
    print("Initial state:")
    print(env.reset(seed=42))
    print()
    
    # Play a simple game
    for step in range(5):
        legal_actions = env.state.legal_actions(env.current_player)
        if not legal_actions:
            break
        action = legal_actions[0]  # Take first legal action
        obs, reward, done, info = env.step(action)
        print(f"Step {step + 1}: Action {action}")
        print(obs)
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        print()
        if done:
            break
