# Tool Parsers

Tool parsers extract tool calls from model responses and generate tool prompts for different model formats.

## Usage

```python
from pettingllms.parser import get_tool_parser

# Get a specific parser
parser = get_tool_parser("r1")

# Parse tool calls from model response
tool_calls = parser.parse(model_response)

# Get tool prompt
prompt = parser.get_tool_prompt(tools_schema)
```

::: pettingllms.parser.tool_parser 