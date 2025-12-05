"""
Few-shot examples for code-related workflow generation.

This module contains example workflow code patterns for code generation,
debugging, refactoring, and other code-related tasks.
"""

# Few-shot examples for different code-related tasks
CODE_WORKFLOW_EXAMPLES = {
    "basic_code_generation": {
        "category": "Basic Code Generation",
        "description": "Single agent generates code based on requirements",
        "code": '''# Example 1: Basic Code Generation
from workflow import AgentNode, Workflow, ToolRegistry
from utils.environments.code_env import CodeEnvironment

# Setup code tools
code_env = CodeEnvironment()
tool_registry = ToolRegistry()

tool_registry.register(
    name="execute_code",
    func=code_env.execute,
    description="Execute Python code and return the output.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute"},
            "timeout": {"type": "integer", "description": "Optional: Execution timeout in seconds (default: 30)"}
        },
        "required": ["code"]
    }
)

tool_registry.register(
    name="read_file",
    func=code_env.read_file,
    description="Read content from a file.",
    parameters={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["filepath"]
    }
)

tool_registry.register(
    name="write_file",
    func=code_env.write_file,
    description="Write content to a file.",
    parameters={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content to write to the file"}
        },
        "required": ["filepath", "content"]
    }
)

# Create a code generation agent
code_agent = AgentNode(
    name="CodeGenerationAgent",
    system_prompt=(
        "You are an expert programmer specializing in Python development.\\n\\n"
        
        "IMPORTANT - YOU MUST USE TOOLS:\\n"
        "You have access to tools that you MUST use for code execution and file operations.\\n\\n"
        
        "RESPONSE FORMAT:\\n"
        "When you need to use a tool, respond in this format:\\n"
        "<think>Your reasoning about what code to write or what action to take</think>\\n"
        '<tool_call>{"name": "tool_name", "parameters": {"param": "value"}}</tool_call>\\n\\n'
        
        "AVAILABLE TOOLS:\\n"
        "1. execute_code(code, timeout): Execute Python code and see the output\\n"
        "2. read_file(filepath): Read content from a file\\n"
        "3. write_file(filepath, content): Write content to a file\\n\\n"
        
        "EXAMPLE:\\n"
        "Human: Write a function to calculate factorial\\n"
        "Assistant: <think>I need to write a factorial function and test it.</think>\\n"
        '<tool_call>{"name": "execute_code", "parameters": {"code": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)\\n\\nprint(factorial(5))"}}</tool_call>\\n'
        "[Tool returns: 120]\\n"
        "Assistant: <think>The function works correctly. Let me save it to a file.</think>\\n"
        '<tool_call>{"name": "write_file", "parameters": {"filepath": "factorial.py", "content": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)"}}</tool_call>\\n'
        "I've created a factorial function that calculates factorials recursively.\\n\\n"
        
        "WORKFLOW:\\n"
        "1. UNDERSTAND the requirements\\n"
        "2. THINK about the solution approach\\n"
        "3. WRITE the code\\n"
        "4. TEST using execute_code\\n"
        "5. REFINE if needed\\n"
        "6. SAVE to file if requested\\n\\n"
        
        "Remember: Always test your code before delivering!"
    ),
    tool_registry=tool_registry,
    max_turns=8,
    enable_conversation_logging=True
)

# Create workflow
workflow = Workflow(name="basic_code_generation")
workflow.add_node(code_agent)

# Run workflow
result = workflow.run(task_description)
print(result.content)
'''
    },
    
    "ensemble_code_generation": {
        "category": "Ensemble Code Generation",
        "description": "Multiple agents generate code with different approaches, then reach consensus",
        "code": '''# Example 2: Ensemble Code Generation
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import EnsembleNode
from utils.environments.code_env import CodeEnvironment

# Setup code tools
code_env = CodeEnvironment()
tool_registry = ToolRegistry()

tool_registry.register(
    name="execute_code",
    func=code_env.execute,
    description="Execute Python code and return the output.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute"}
        },
        "required": ["code"]
    }
)

# Create multiple coding agents with different styles
agent1 = AgentNode(
    name="OptimizationAgent",
    system_prompt=(
        "You are a performance-focused programmer. Write efficient, optimized code "
        "with good time and space complexity. Focus on algorithm efficiency."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

agent2 = AgentNode(
    name="ReadabilityAgent",
    system_prompt=(
        "You are a clean code advocate. Write readable, maintainable code with "
        "clear variable names, good structure, and comprehensive documentation."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

agent3 = AgentNode(
    name="RobustnessAgent",
    system_prompt=(
        "You are a defensive programmer. Write robust code with proper error handling, "
        "input validation, and edge case coverage."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

# Create synthesis agent
synthesis_agent = AgentNode(
    name="SynthesisAgent",
    system_prompt=(
        "You are a senior software architect. Review multiple code implementations "
        "and create the best solution that balances performance, readability, and robustness."
    ),
    tool_registry=tool_registry,
    max_turns=4
)

# Create ensemble node
ensemble = EnsembleNode(
    name="CodeEnsemble",
    agents=[agent1, agent2, agent3],
    strategy="consensus",
    consensus_agent=synthesis_agent
)

# Create workflow
workflow = Workflow(name="ensemble_code_generation")
workflow.add_node(ensemble)

# Run workflow
result = workflow.run(task_description)
print(result.content)
'''
    },
    
    "debug_workflow": {
        "category": "Conditional Debugging Workflow",
        "description": "Graph workflow that routes to different fixers based on error type",
        "code": '''# Example 3: Conditional Multi-Agent Debugging
from workflow import AgentNode, AgentGraph, ToolRegistry
from utils.environments.code_env import CodeEnvironment

# Setup code tools
code_env = CodeEnvironment()
tool_registry = ToolRegistry()

tool_registry.register(
    name="execute_code",
    func=code_env.execute,
    description="Execute Python code and return the output.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute"}
        },
        "required": ["code"]
    }
)

tool_registry.register(
    name="read_file",
    func=code_env.read_file,
    description="Read content from a file.",
    parameters={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["filepath"]
    }
)

# Stage 1: Error classifier
classifier = AgentNode(
    name="ErrorClassifier",
    system_prompt=(
        "You are a debugging expert. Analyze the buggy code and classify the error type.\\n"
        "Reply with ONE word:\\n"
        "- SYNTAX: for syntax errors\\n"
        "- LOGIC: for logic errors\\n"
        "- RUNTIME: for runtime errors\\n"
        "- PERFORMANCE: for performance issues\\n\\n"
        "Store the error type in context state 'error_type'."
    ),
    tool_registry=tool_registry,
    max_turns=3
)

# Specialized fixers for different error types
syntax_fixer = AgentNode(
    name="SyntaxFixer",
    system_prompt=(
        "You are a syntax error specialist. Fix syntax errors including:\\n"
        "- Missing colons, brackets, quotes\\n"
        "- Indentation issues\\n"
        "- Invalid syntax patterns\\n"
        "Test the fixed code."
    ),
    tool_registry=tool_registry,
    max_turns=5
)

logic_fixer = AgentNode(
    name="LogicFixer",
    system_prompt=(
        "You are a logic error specialist. Fix logic errors including:\\n"
        "- Incorrect algorithms\\n"
        "- Wrong conditions\\n"
        "- Off-by-one errors\\n"
        "Test with multiple cases."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

runtime_fixer = AgentNode(
    name="RuntimeFixer",
    system_prompt=(
        "You are a runtime error specialist. Fix runtime errors including:\\n"
        "- Null pointer exceptions\\n"
        "- Type errors\\n"
        "- Index out of bounds\\n"
        "Add proper error handling."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

performance_optimizer = AgentNode(
    name="PerformanceOptimizer",
    system_prompt=(
        "You are a performance optimization specialist. Fix performance issues:\\n"
        "- Inefficient algorithms\\n"
        "- Memory leaks\\n"
        "- Unnecessary computations\\n"
        "Benchmark before and after."
    ),
    tool_registry=tool_registry,
    max_turns=7
)

# Verifier
verifier = AgentNode(
    name="Verifier",
    system_prompt=(
        "You are a QA engineer. Verify the fix:\\n"
        "1. Run comprehensive tests\\n"
        "2. Check edge cases\\n"
        "3. Ensure no regressions\\n"
        "Set context state 'fix_verified' to True/False."
    ),
    tool_registry=tool_registry,
    max_turns=5
)

# Create graph with conditional routing
graph = AgentGraph(max_steps=25)
graph.add_node("classifier", classifier)
graph.add_node("syntax_fixer", syntax_fixer)
graph.add_node("logic_fixer", logic_fixer)
graph.add_node("runtime_fixer", runtime_fixer)
graph.add_node("performance_optimizer", performance_optimizer)
graph.add_node("verifier", verifier)

# Route based on error type
def route_by_error_type(context):
    """Route to appropriate fixer based on error classification"""
    error_type = context.get_state("error_type", "")
    msg_content = context.get_latest_message().content.upper()
    
    if "SYNTAX" in error_type or "SYNTAX" in msg_content:
        return "syntax_fixer"
    elif "LOGIC" in error_type or "LOGIC" in msg_content:
        return "logic_fixer"
    elif "RUNTIME" in error_type or "RUNTIME" in msg_content:
        return "runtime_fixer"
    elif "PERFORMANCE" in error_type or "PERFORMANCE" in msg_content:
        return "performance_optimizer"
    else:
        return "logic_fixer"  # default

graph.add_conditional_edges("classifier", route_by_error_type)

# All fixers go to verifier
graph.add_edge("syntax_fixer", "verifier")
graph.add_edge("logic_fixer", "verifier")
graph.add_edge("runtime_fixer", "verifier")
graph.add_edge("performance_optimizer", "verifier")

# After verification, retry if not verified (max 2 attempts)
def route_after_verification(context):
    """Re-classify if verification failed"""
    fix_verified = context.get_state("fix_verified", False)
    retry_count = context.get_state("retry_count", 0)
    
    if fix_verified:
        return None  # Done, exit
    elif retry_count < 1:
        context.set_state("retry_count", retry_count + 1)
        return "classifier"  # Try again
    else:
        return None  # Give up after 1 retry

# Set entry and finish
graph.set_entry_point("classifier")
graph.set_finish_point("verifier")

# Run graph
result = graph.run(buggy_code)
print(result.content)
'''
    },
    
    "refactoring_workflow": {
        "category": "Code Refactoring",
        "description": "Agent analyzes code and suggests/applies refactoring improvements",
        "code": '''# Example 4: Code Refactoring Workflow
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import ReflectionNode
from utils.environments.code_env import CodeEnvironment

# Setup code tools
code_env = CodeEnvironment()
tool_registry = ToolRegistry()

tool_registry.register(
    name="execute_code",
    func=code_env.execute,
    description="Execute Python code and return the output.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute"}
        },
        "required": ["code"]
    }
)

tool_registry.register(
    name="read_file",
    func=code_env.read_file,
    description="Read content from a file.",
    parameters={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["filepath"]
    }
)

# Create refactoring agent
refactor_agent = AgentNode(
    name="RefactoringAgent",
    system_prompt=(
        "You are a code refactoring expert. Analyze code and improve it by:\\n"
        "1. Removing code smells\\n"
        "2. Improving naming and structure\\n"
        "3. Extracting reusable functions\\n"
        "4. Applying design patterns where appropriate\\n"
        "5. Ensuring the refactored code maintains the same functionality\\n\\n"
        "Always test after refactoring to ensure correctness."
    ),
    tool_registry=tool_registry,
    max_turns=8
)

# Create reflection node for iterative improvement
reflection = ReflectionNode(
    name="RefactoringReflection",
    agent=refactor_agent,
    num_iterations=2
)

# Create workflow
workflow = Workflow(name="refactoring_workflow")
workflow.add_node(reflection)

# Run workflow
result = workflow.run(code_to_refactor)
print(result.content)
'''
    },
    
    "code_review_workflow": {
        "category": "Conditional Code Review",
        "description": "Graph workflow that routes based on code quality checks",
        "code": '''# Example 5: Code Review Workflow
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import DebateNode
from utils.environments.code_env import CodeEnvironment

# Setup code tools
code_env = CodeEnvironment()
tool_registry = ToolRegistry()

tool_registry.register(
    name="read_file",
    func=code_env.read_file,
    description="Read content from a file.",
    parameters={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["filepath"]
    }
)

tool_registry.register(
    name="execute_code",
    func=code_env.execute,
    description="Execute Python code and return the output.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute"}
        },
        "required": ["code"]
    }
)

# Initial quality checker
quality_checker = AgentNode(
    name="QualityChecker",
    system_prompt=(
        "You are a code quality analyst. Perform initial assessment:\\n"
        "1. Check for obvious issues\\n"
        "2. Rate code quality: EXCELLENT, GOOD, NEEDS_REVIEW, CRITICAL\\n"
        "3. Identify which aspects need deep review (security/performance/maintainability)\\n\\n"
        "Store quality rating in context state 'quality_rating'."
    ),
    tool_registry=tool_registry,
    max_turns=4
)

# Create specialized reviewers
security_reviewer = AgentNode(
    name="SecurityReviewer",
    system_prompt=(
        "You are a security expert. Deep dive into security issues:\\n"
        "- Input validation\\n"
        "- SQL injection risks\\n"
        "- Authentication/authorization\\n"
        "- Data encryption\\n"
        "Set context state 'security_issues' with severity level."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

performance_reviewer = AgentNode(
    name="PerformanceReviewer",
    system_prompt=(
        "You are a performance expert. Analyze performance:\\n"
        "- Time/space complexity\\n"
        "- Database query optimization\\n"
        "- Caching opportunities\\n"
        "- Bottlenecks\\n"
        "Set context state 'performance_issues' with severity level."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

maintainability_reviewer = AgentNode(
    name="MaintainabilityReviewer",
    system_prompt=(
        "You are a code quality expert. Evaluate maintainability:\\n"
        "- Code structure and organization\\n"
        "- Documentation\\n"
        "- Best practices\\n"
        "- Test coverage\\n"
        "Set context state 'maintainability_issues' with severity level."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

# Refactoring agent (called if issues found)
refactoring_agent = AgentNode(
    name="RefactoringAgent",
    system_prompt=(
        "You are a refactoring expert. Based on review feedback:\\n"
        "1. Address critical issues\\n"
        "2. Apply necessary refactoring\\n"
        "3. Improve code quality\\n"
        "4. Re-run quality checks"
    ),
    tool_registry=tool_registry,
    max_turns=8
)

# Final approval agent
approval_agent = AgentNode(
    name="ApprovalAgent",
    system_prompt=(
        "You are a senior architect. Make final approval decision:\\n"
        "- APPROVED: code is good to merge\\n"
        "- APPROVED_WITH_COMMENTS: acceptable with minor notes\\n"
        "- CHANGES_REQUESTED: needs improvements\\n"
        "Provide clear summary and action items."
    ),
    tool_registry=tool_registry,
    max_turns=3
)

# Create graph with conditional routing
graph = AgentGraph(max_steps=30)
graph.add_node("quality_checker", quality_checker)
graph.add_node("security_reviewer", security_reviewer)
graph.add_node("performance_reviewer", performance_reviewer)
graph.add_node("maintainability_reviewer", maintainability_reviewer)
graph.add_node("refactoring_agent", refactoring_agent)
graph.add_node("approval_agent", approval_agent)

# Route based on quality rating
def route_after_quality_check(context):
    """Route to appropriate reviewers based on quality rating"""
    quality_rating = context.get_state("quality_rating", "").upper()
    msg_content = context.get_latest_message().content.upper()
    
    if "EXCELLENT" in quality_rating or "EXCELLENT" in msg_content:
        return "approval_agent"  # Skip deep review
    elif "CRITICAL" in quality_rating or "CRITICAL" in msg_content:
        return "security_reviewer"  # Start with security
    else:
        return "maintainability_reviewer"  # Start with maintainability

graph.add_conditional_edges("quality_checker", route_after_quality_check)

# Chain reviews
graph.add_edge("security_reviewer", "performance_reviewer")
graph.add_edge("performance_reviewer", "maintainability_reviewer")
graph.add_edge("maintainability_reviewer", "approval_agent")

# After approval, check if refactoring needed
def route_after_approval(context):
    """Route to refactoring if changes requested"""
    approval_msg = context.get_latest_message().content.upper()
    refactor_count = context.get_state("refactor_count", 0)
    
    if "CHANGES_REQUESTED" in approval_msg and refactor_count < 2:
        context.set_state("refactor_count", refactor_count + 1)
        return "refactoring_agent"
    else:
        return None  # Done

graph.add_conditional_edges("approval_agent", route_after_approval)

# After refactoring, go back to quality check
graph.add_edge("refactoring_agent", "quality_checker")

# Set entry and finish
graph.set_entry_point("quality_checker")
graph.set_finish_point("approval_agent")

# Run graph
result = graph.run(code_to_review)
print(result.content)
'''
    },
    
    "test_generation_workflow": {
        "category": "Test Generation with Conditional Fix",
        "description": "Graph workflow: analyze -> generate tests -> verify -> fix if failed",
        "code": '''# Example 6: Test Generation with Conditional Verification
from workflow import AgentNode, AgentGraph, ToolRegistry
from utils.environments.code_env import CodeEnvironment

# Setup code tools
code_env = CodeEnvironment()
tool_registry = ToolRegistry()

tool_registry.register(
    name="read_file",
    func=code_env.read_file,
    description="Read content from a file.",
    parameters={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["filepath"]
    }
)

tool_registry.register(
    name="execute_code",
    func=code_env.execute,
    description="Execute Python code and return the output.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute"}
        },
        "required": ["code"]
    }
)

tool_registry.register(
    name="write_file",
    func=code_env.write_file,
    description="Write content to a file.",
    parameters={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content to write to the file"}
        },
        "required": ["filepath", "content"]
    }
)

# Stage 1: Code analyzer
analyzer = AgentNode(
    name="CodeAnalyzer",
    system_prompt=(
        "You are a code analysis expert. Analyze the given code and identify:\\n"
        "1. All functions and their purposes\\n"
        "2. Input/output specifications\\n"
        "3. Edge cases to test\\n"
        "4. Error conditions to handle"
    ),
    tool_registry=tool_registry,
    max_turns=4
)

# Stage 2: Test generator
test_generator = AgentNode(
    name="TestGenerator",
    system_prompt=(
        "You are a test automation expert. Based on the code analysis, generate "
        "comprehensive unit tests using pytest. Cover:\\n"
        "1. Normal cases\\n"
        "2. Edge cases\\n"
        "3. Error conditions\\n"
        "4. Boundary values\\n\\n"
        "Execute the tests and set context state 'tests_passed' to True/False."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

# Stage 3: Test verifier (runs when tests pass)
verifier = AgentNode(
    name="TestVerifier",
    system_prompt=(
        "You are a QA specialist. Verify that:\\n"
        "1. All tests pass\\n"
        "2. Code coverage is adequate\\n"
        "3. Tests are meaningful and correct\\n"
        "Provide final approval."
    ),
    tool_registry=tool_registry,
    max_turns=3
)

# Stage 4: Code fixer (runs when tests fail)
code_fixer = AgentNode(
    name="CodeFixer",
    system_prompt=(
        "You are a code repair expert. The tests failed. Analyze the failures and:\\n"
        "1. Identify why tests are failing\\n"
        "2. Fix the code to pass all tests\\n"
        "3. Re-run tests to verify fixes\\n"
        "Set context state 'tests_passed' after fixing."
    ),
    tool_registry=tool_registry,
    max_turns=8
)

# Create graph with conditional routing
graph = AgentGraph(max_steps=20)
graph.add_node("analyzer", analyzer)
graph.add_node("test_generator", test_generator)
graph.add_node("verifier", verifier)
graph.add_node("code_fixer", code_fixer)

# Sequential edges
graph.add_edge("analyzer", "test_generator")

# Conditional routing based on test results
def route_after_test(context):
    """Route to verifier if tests pass, otherwise to code_fixer"""
    tests_passed = context.get_state("tests_passed", False)
    latest_msg = context.get_latest_message().content
    
    # Check if tests passed in the message
    if tests_passed or "all tests passed" in latest_msg.lower() or "0 failed" in latest_msg.lower():
        context.set_state("tests_passed", True)
        return "verifier"
    else:
        context.set_state("tests_passed", False)
        return "code_fixer"

graph.add_conditional_edges("test_generator", route_after_test)

# After fixing, go back to test_generator or to verifier
def route_after_fix(context):
    """After fixing, check if tests now pass"""
    tests_passed = context.get_state("tests_passed", False)
    fix_attempts = context.get_state("fix_attempts", 0)
    
    if tests_passed:
        return "verifier"
    elif fix_attempts < 2:  # Max 2 fix attempts
        context.set_state("fix_attempts", fix_attempts + 1)
        return "test_generator"
    else:
        return "verifier"  # Give up and verify what we have

graph.add_conditional_edges("code_fixer", route_after_fix)

# Set entry and finish points
graph.set_entry_point("analyzer")
graph.set_finish_point("verifier")

# Run graph
result = graph.run(code_to_test)
print(result.content)
'''
    },
    
    "code_migration_workflow": {
        "category": "Code Migration",
        "description": "Graph workflow with conditional routing for different migration tasks",
        "code": '''# Example 7: Code Migration with Conditional Routing
from workflow import AgentNode, AgentGraph, ToolRegistry
from utils.environments.code_env import CodeEnvironment

# Setup code tools
code_env = CodeEnvironment()
tool_registry = ToolRegistry()

tool_registry.register(
    name="read_file",
    func=code_env.read_file,
    description="Read content from a file.",
    parameters={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["filepath"]
    }
)

tool_registry.register(
    name="execute_code",
    func=code_env.execute,
    description="Execute Python code and return the output.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute"}
        },
        "required": ["code"]
    }
)

tool_registry.register(
    name="write_file",
    func=code_env.write_file,
    description="Write content to a file.",
    parameters={
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content to write to the file"}
        },
        "required": ["filepath", "content"]
    }
)

# Create router agent
router = AgentNode(
    name="MigrationRouter",
    system_prompt=(
        "Analyze the migration task and determine the type. "
        "Reply ONLY with one word: 'LANGUAGE' for language migration, "
        "'VERSION' for version upgrade, or 'FRAMEWORK' for framework migration."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Create specialized migration agents
language_migrator = AgentNode(
    name="LanguageMigrator",
    system_prompt=(
        "You are a language migration expert. Convert code from one programming "
        "language to another while preserving functionality and idioms."
    ),
    tool_registry=tool_registry,
    max_turns=8
)

version_upgrader = AgentNode(
    name="VersionUpgrader",
    system_prompt=(
        "You are a version upgrade specialist. Update code to work with newer "
        "versions of the language or libraries, handling deprecated features."
    ),
    tool_registry=tool_registry,
    max_turns=8
)

framework_migrator = AgentNode(
    name="FrameworkMigrator",
    system_prompt=(
        "You are a framework migration expert. Migrate code from one framework "
        "to another, adapting patterns and best practices."
    ),
    tool_registry=tool_registry,
    max_turns=8
)

# Create graph
graph = AgentGraph(max_steps=15)
graph.add_node("router", router)
graph.add_node("language", language_migrator)
graph.add_node("version", version_upgrader)
graph.add_node("framework", framework_migrator)

# Define routing logic
def route_by_migration_type(context):
    decision = context.get_latest_message().content.upper()
    if "LANGUAGE" in decision:
        return "language"
    elif "VERSION" in decision:
        return "version"
    elif "FRAMEWORK" in decision:
        return "framework"
    return "version"  # default

graph.add_conditional_edges("router", route_by_migration_type)

# Set entry and finish points
graph.set_entry_point("router")
graph.set_finish_point("language")
graph.set_finish_point("version")
graph.set_finish_point("framework")

# Run graph
result = graph.run(migration_task)
print(result.content)
'''
    }
}


# Prompt template for code-related task generation
CODE_GENERATION_PROMPT_TEMPLATE = """You are an expert software engineer specializing in multi-agent code development systems. Your task is to generate workflow code for code-related tasks based on the user's requirements.

## Available Workflow Patterns

You have access to the following workflow patterns for code-related tasks:

1. **basic_code_generation**: Single agent generates code (for straightforward coding tasks)
2. **ensemble_code_generation**: Multiple agents generate code with different approaches (for complex algorithms)
3. **debug_workflow**: **CONDITIONAL GRAPH** that routes to specialized fixers based on error type (syntax/logic/runtime/performance)
4. **refactoring_workflow**: Iterative code improvement with reflection (for refactoring and cleaning up code)
5. **code_review_workflow**: **CONDITIONAL GRAPH** that routes based on quality checks and may trigger refactoring
6. **test_generation_workflow**: **CONDITIONAL GRAPH** that automatically fixes code if tests fail, with retry loop
7. **code_migration_workflow**: **CONDITIONAL GRAPH** routing for different migration tasks (language/version/framework)

**IMPORTANT**: Patterns 3, 5, 6, 7 use AgentGraph with conditional routing. PREFER these for complex tasks requiring dynamic decision-making.

## Few-Shot Examples

{examples}

## Task

Given:
- **Task Category**: {category}
- **Specific Task**: {task}

Generate complete, runnable Python code that:
1. Imports all necessary modules at the top
2. Sets up the appropriate workflow pattern
3. Configures agents with suitable system prompts for the task
4. Runs the workflow and produces the result
5. Is self-contained and can be executed directly

## Requirements

1. Start with all imports:
```python
import os
import sys
# Add more imports as needed
```

2. Use the workflow pattern that best fits the task
3. Design agents that are specific to the code task at hand
4. Include proper code execution and testing
5. Make sure the code is complete and runnable

## Output Format

IMPORTANT: You must first reason about the workflow design in a <think> block, then provide the code in a <code> block.

In your <think> block, you MUST answer these questions:
1. **Task Analysis**: What is this task asking? What kind of code work is needed?
2. **Workflow Pattern Selection**: Which workflow pattern is most suitable and why?
   - Is this a simple code generation task? → basic_code_generation
   - Does it need multiple approaches? → ensemble_code_generation
   - Is it fixing bugs WITH CONDITIONAL routing by error type? → debug_workflow (AgentGraph)
   - Is it improving existing code? → refactoring_workflow
   - Is it reviewing code WITH CONDITIONAL quality checks? → code_review_workflow (AgentGraph)
   - Is it generating tests WITH AUTOMATIC fixing if tests fail? → test_generation_workflow (AgentGraph)
   - Is it migrating code WITH CONDITIONAL routing by migration type? → code_migration_workflow (AgentGraph)
3. **Conditional Logic**: What conditions determine the workflow path? (e.g., test pass/fail, error type, quality rating)
4. **Agent Design**: What agents are needed? What specialized agents handle different branches?
5. **Tool Requirements**: What tools do the agents need? (execute_code, read_file, write_file, etc.)
6. **Graph Structure**: How are nodes connected? What are the conditional edges and routing functions?
7. **Loop/Retry Logic**: Are there retry loops? What are the exit conditions?

Format:
<think>
1. Task Analysis: [Your analysis of what the task needs]
2. Workflow Pattern: [Selected pattern and justification]
3. Agent Design: [Description of agents needed and their roles]
4. Tool Requirements: [What tools are needed]
5. Workflow Flow: [How agents will interact]
</think>
<code>
```python
...
```
</code>
"""


def get_code_generation_prompt(task: str, include_examples: list = None,
                               use_simple_format: bool = False, random_sample_examples: bool = True,
                               force_nested: bool = False) -> tuple:
    """
    Generate a prompt for code-related task generation.

    Args:
        task: The specific code task to accomplish
        include_examples: List of example categories to include (default: randomly sampled)
        use_simple_format: If True, return simple format without detailed guide (for SFT data)
        random_sample_examples: If True, randomly sample examples for diversity (default: True)
        force_nested: If True, instruct LLM to create nested/combined workflows

    Returns:
        Tuple of (prompt_string, selected_examples)
    """
    import random

    # Simple format for SFT data collection (just task description)
    if use_simple_format:
        return f"Task: {task}", []

    # Determine which examples to include
    if include_examples is None:
        all_examples = list(CODE_WORKFLOW_EXAMPLES.keys())

        # Define sampling weights: basic gets 15% probability, others share 85%
        weights = []
        for ex in all_examples:
            if ex == "basic_code_generation":
                weights.append(0.15)  # 15% weight for basic
            else:
                weights.append(0.85 / (len(all_examples) - 1))  # Share remaining 85%

        # Sample with weighted probabilities
        num_to_sample = max(2, len(all_examples) // 2)  # At least 2 examples
        include_examples = random.choices(all_examples, weights=weights, k=num_to_sample)

    # Build examples section
    examples_text = ""
    for ex_name in include_examples:
        if ex_name in CODE_WORKFLOW_EXAMPLES:
            ex = CODE_WORKFLOW_EXAMPLES[ex_name]
            examples_text += f"\n### {ex['category']}\n"
            examples_text += f"**Description**: {ex['description']}\n\n"
            examples_text += f"```python\n{ex['code']}\n```\n"

    # Add instruction for nested/combined workflows if requested
    nested_instruction = ""
    if force_nested:
        nested_instruction = """

IMPORTANT: For this task, create a NESTED/COMBINED workflow by combining multiple patterns.
For example:
- Use ensemble code generation where each agent uses reflection
- Use debug workflow where each stage uses ensemble
- Use code review where each reviewer uses complex analysis process
- Combine migration workflow with any of the above patterns

Be creative and create a sophisticated multi-layered workflow that combines 2-3 patterns."""

    # Format the prompt (no category specified - let LLM decide from examples)
    prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(
        examples=examples_text,
        category="auto-detect from task",
        task=task
    ) + nested_instruction

    return prompt, include_examples


# Task category selection prompt
TASK_CATEGORY_SELECTION_PROMPT = """Given a code-related task, determine the most appropriate workflow pattern.

Available patterns:
- **basic_code_generation**: ONLY for simple, straightforward code generation (e.g., "Write a function to sort a list"). Use sparingly - less than 15% of tasks.
- **ensemble_code_generation**: For complex algorithms that benefit from multiple approaches and perspectives. Use for most non-trivial code generation tasks.
- **debug_workflow**: For fixing bugs or errors in existing code. Use when the task involves debugging or error resolution.
- **refactoring_workflow**: For improving, restructuring, or optimizing existing code. PREFER this for code improvement tasks.
- **code_review_workflow**: For analyzing and reviewing code quality, security, and best practices.
- **test_generation_workflow**: For creating unit tests, integration tests, or test suites. Use for testing-related tasks.
- **code_migration_workflow**: For converting code between languages, versions, or frameworks. Use for migration/conversion tasks.

IMPORTANT GUIDELINES:
- AVOID basic_code_generation unless the task is trivially simple
- PREFER conditional graph workflows (debug_workflow, code_review_workflow, test_generation_workflow) for tasks requiring:
  * Dynamic decision-making based on results
  * Automatic fixing/retry logic
  * Different handling for different error types or conditions
- Use debug_workflow for bug fixing WITH conditional routing by error type
- Use code_review_workflow for analysis WITH conditional quality checks and auto-refactoring
- Use test_generation_workflow for testing WITH automatic code fixing if tests fail
- Use code_migration_workflow for conversion WITH conditional routing by migration type
- Use ensemble_code_generation for complex algorithms needing multiple perspectives
- Use refactoring_workflow for iterative improvement with reflection

Task: {task}

Analyze the task complexity and reply with ONLY the pattern name (one of: basic_code_generation, ensemble_code_generation, debug_workflow, refactoring_workflow, code_review_workflow, test_generation_workflow, code_migration_workflow).
"""


def get_task_category_selection_prompt(task: str) -> str:
    """Get prompt for selecting the appropriate workflow category for code tasks."""
    return TASK_CATEGORY_SELECTION_PROMPT.format(task=task)
