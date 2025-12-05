from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple, Union
import json

class AIClient:
    def __init__(self, api_base: str, api_key: str, chat_model: str, max_answer_tokens: int):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.chat_model = chat_model
        self.max_answer_tokens = max_answer_tokens

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.2, max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, int, int]:
        # Guard against over-large max_tokens which can shrink available input window
        # and trigger context length errors even on the first turn.
        configured = int(max_tokens or self.max_answer_tokens)
        # Cap to a safe upper bound to keep total within typical context windows
        max_tokens = int(min(configured, 8192))
        
        # Build request parameters
        params = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add tools if provided
        if tools:
            params["tools"] = tools
            # Note: tool_choice removed - DeepSeek API doesn't support it
        
        resp = self.client.chat.completions.create(**params)
        usage = resp.usage
        message = resp.choices[0].message
        
        # Handle OpenAI native function calling format
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Model wants to call tools - convert to custom format
            tool_calls_list = []
            for tool_call in message.tool_calls:
                tool_calls_list.append({
                    "name": tool_call.function.name,
                    "parameters": json.loads(tool_call.function.arguments)
                })
            
            # If multiple tool calls, just use the first one for now
            tool_call_info = tool_calls_list[0]
            
            # Format as custom tool call string
            content = f'<tool_call>{json.dumps(tool_call_info)}</tool_call>'
            
            # Add any text content if present
            if message.content:
                content = message.content + "\n" + content
        else:
            # Normal text response
            content = message.content if message.content is not None else ""
        
        return content, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)

