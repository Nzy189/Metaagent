from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class TicTacToeEnvConfig:
    visual_obs: bool = True
    image_dir: Optional[str] = None
    recording_type: str = 'gif'
    recording_fps: int = 2
    render_mode: str = "text"
    llm_play_mode: str="self_play"
    action_lookup: Optional[Dict[int, Dict[int, str]]] = field(default_factory=lambda: {
        0: {0: "X(0,0)", 1: "X(0,1)", 2: "X(0,2)",
            3: "X(1,0)", 4: "X(1,1)", 5: "X(1,2)",
            6: "X(2,0)", 7: "X(2,1)", 8: "X(2,2)"},
        1: {0: "O(0,0)", 1: "O(0,1)", 2: "O(0,2)",
            3: "O(1,0)", 4: "O(1,1)", 5: "O(1,2)",
            6: "O(2,0)", 7: "O(2,1)", 8: "O(2,2)"}
    })
    grid_lookup: Optional[Dict[str, str]] = field(default_factory=lambda: {
        ".": "empty", "x": "X", "o": "O"
    }) 