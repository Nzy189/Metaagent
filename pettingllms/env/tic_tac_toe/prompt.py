import re
from textwrap import dedent
import dataclasses
from typing import Any, List
import random

@dataclasses.dataclass
class Render:
    text: str
    image_paths: List[str]

@dataclasses.dataclass
class Observations:
    is_terminal: bool
    agent_id: int
    reward: dict[int,float]
    legal_actions: list[int]
    action_lookup: dict[int, str]
    state_render: Render







system_prompt = ("You are an AI agent that makes optimal decisions in the game of tic-tac-toe.")
rules_prompt = ("1. Tic-tac-toe is a two-player board game played on a three-by-three grid. "
         "The grid is 0-indexed, where (0,0) is the top-left corner and (2,2) is the bottom-right corner.\n"
         "2. Two players take turns placing their marks X and O in empty cells of the grid.\n"
         "3. The marks . represent empty cells.\n"
         "4. The player who first places three of their marks in a horizontal, vertical, or diagonal line wins.\n"
         "5. If all cells are filled and no player wins, the game ends in a draw.")

class RegexError(Exception):
    """Raised when the response is illegal"""
    pass


 



def extract_action(agent_id, agent_response,legal_actions):
    # Optimized regex patterns, sorted by priority
    regex_patterns = [
        # 1. Match complete JSON code block (supports multi-line)
        (r'```json\s*\{\s*"action"\s*:\s*"([^"]+)"\s*\}\s*```', lambda m: m.strip()),
        
        # 2. Match action field in JSON (no code block wrapper needed)
        (r'"action"\s*:\s*"([^"]+)"', lambda m: m.strip()),
        
        # 3. Direct match of X(i,j) or O(i,j) format
        (r'([XOxo])\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', 
         lambda m: f"{m[0].upper()}({m[1]},{m[2]})"),
        
        # 4. Match (i,j) format, add corresponding mark based on agent_id
        (r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', 
         lambda m: f"{'X' if agent_id == 0 else 'O'}({m[0]},{m[1]})")
    ]
    
    for pattern, processor in regex_patterns:
        match = re.findall(pattern, agent_response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            
            action_str = processor(match[-1])
            # Convert string format to 0-8 integer action
            action_int = _convert_action_str_to_int(action_str)
            if action_int in legal_actions:
                return action_int
            else:
                continue
    print(f"Error response: {agent_response}")
    return random.choice(legal_actions)

def _convert_action_str_to_int(action_str):
    """Convert action string to 0-8 integer"""
    # Match format like "X(1,2)" or "O(0,1)"
    match = re.match(r'[XOxo]\((\d+),(\d+)\)', action_str.strip())
    if match:
        row, col = int(match.group(1)), int(match.group(2))
        # Validate coordinate range
        if 0 <= row <= 2 and 0 <= col <= 2:
            # Convert (row, col) to 0-8 index
            return row * 3 + col
    
    # If already a number, return directly
    try:
        action_int = int(action_str)
        if 0 <= action_int <= 8:
            return action_int
    except ValueError:
        print(ValueError(f"Invalid action format: {action_str}"))
        return random.choice(range(9))

def get_env_background_prompt():
    return f"""
    {system_prompt}
    The rules of the game are:
    {rules_prompt}
    """




def get_action_prompt(agent_id,legal_actions,action_lookup):
    mark = 'X' if agent_id == 0 else 'O'
    legal_actions_str = [action_lookup[agent_id][action] for action in legal_actions]
    action_prompt = dedent(f"""\
        INSTRUCTIONS:
        Your mark is {mark}.
        Now it is your turn to choose an action. You should according to the current state of the game and the legal actions answer with the reasoning and your action in the format: "{mark}(i,j)"
        where i is the row index and j is the column index. 
        The legal actions means the actions that you can take in the current state of the game. You can not choose an action that is not in the legal actions or the cell is already taken.
        The legal actions are:
        {legal_actions_str}. 
        Based on the available information, analyze the situation and provide your thought process and next move in the format: "{mark}(i,j)".
        Your response should be in the following format:
        ```json
        {{
            "reasoning": "your reasoning",
            "action": "{mark}(i,j)"
        }}
        ```
        """)
    return action_prompt


def get_observation_prompt(state_render_text):
    observation_txt_prompt=f"""
    The current state of the game is:
    \n\n
    {state_render_text}"""
   
    return f"""
    {observation_txt_prompt}
    """


def get_prompt_for_self_play(observations:Observations):
    bg_prompt = get_env_background_prompt()
    action_prompt = get_action_prompt(observations.agent_id,observations.legal_actions,observations.action_lookup)
    observation_prompt = get_observation_prompt(observations.state_render.text)
    prompt_render=Render(bg_prompt+action_prompt+observation_prompt,observations.state_render.image_paths)

    return prompt_render

