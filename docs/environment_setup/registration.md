# **Environment and Agent Registration**

The framework uses a centralized registration system to map string identifiers to environment and agent classes. This enables flexible configuration-based instantiation.

---

## **Registration System**

**Location**: `pettingllms/trainer/multiagentssys_register.py`

The registration file defines four dictionaries that map string keys to classes:

1. **ENV_CLASSES**: Single environment instances
2. **ENV_BATCH_CLASSES**: Batched environment managers
3. **AGENT_CLASSES**: Agent implementations
4. **ENV_WORKER_CLASSES**: Ray-based execution workers

---


## **Safe Import Pattern**

The registration uses `safe_import()` to handle missing dependencies gracefully:

```python
def safe_import(module_path, class_name):
    """Import a class from a module, returning None if it fails."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None
```

**Benefits**:
- Missing optional dependencies don't crash the framework
- Only installed environments/agents are available
- Failed imports are filtered out: `{k: v for k, v in ENV_CLASSES.items() if v is not None}`

---

## **Usage in Configuration**

### **Environment Registration**

In your config file:

```yaml
env:
  name: code_env  # Maps to ENV_CLASSES["code_env"] → CodeEnv
  max_turns: 8
```

The execution engine resolves this:

```python
# From: pettingllms/trainer/multi_agents_execution_engine.py
env_name = config.env.name  # "code_env"
self.env_class = ENV_CLASS_MAPPING[env_name]  # CodeEnv
```

### **Agent Registration**

In your config file:

```yaml
multi_agent_interaction:
  turn_order: ["code_generator", "test_generator"]
```

The execution engine resolves this:

```python
# From: pettingllms/trainer/multi_agents_execution_engine.py
self.agent_class_list = [
    AGENT_CLASS_MAPPING["code_generator"],  # CodeGenerationAgent
    AGENT_CLASS_MAPPING["test_generator"]   # UnitTestGenerationAgent
]
```

---

## **⚠️ Critical Requirement: Unique Keys**

**All environment and agent keys MUST be globally unique within their respective registries.**

### **Why Uniqueness Matters**

Python dictionaries only allow one value per key:

```python
❌ WRONG: Duplicate keys cause conflicts
ENV_CLASSES = {
    "code_env": CodeEnvV1,
    "code_env": CodeEnvV2,  # Overwrites previous entry!
}
# Only CodeEnvV2 is accessible

✅ CORRECT: Use unique, descriptive keys
ENV_CLASSES = {
    "code_env": CodeEnv,
    "code_env_single_agent": CodeEnvSingleAgent,
}
```

### **Naming Conventions**

Use descriptive suffixes for variants:

```python
# Task-based suffixes
"code_env"              # Multi-agent code environment
"code_env_single_agent" # Single-agent variant

# Domain-based suffixes
"math_env"              # Standard math environment
"math_aggretion_env"    # Math with aggregation

# Agent role suffixes
"code_generator"        # Code generation agent
"test_generator"        # Test generation agent
```

---

## **Next Steps**

Continue exploring environment setup:

- Review framework architecture: [Core Architecture](core-architecture.md)
- Learn about agent implementation: [Agent Functions](agent-functions.md)
- Understand state management: [Environment State](environment-state.md)

