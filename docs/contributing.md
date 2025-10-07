# Contributing to PettingLLMs

Thank you for your interest in contributing to PettingLLMs!

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists
2. Create a new issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details

### Pull Requests

We welcome pull requests! Follow these steps:

1. **Fork the repository**
   ```bash
   git clone https://github.com/NorahYujieZhao/PettingLLMs.git
   cd PettingLLMs
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, documented code
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   # Run existing tests
   pytest tests/
   
   # Test your feature
   python scripts/test_feature.py
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create pull request**
   - Describe your changes
   - Reference related issues
   - Request review

## Development Setup

### Install Development Dependencies

```bash
# Clone repository
git clone https://github.com/NorahYujieZhao/PettingLLMs.git
cd PettingLLMs

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements_dev.txt  # If available
```

### Code Style

We follow PEP 8 style guidelines:

```bash
# Format code with black
black pettingllms/

# Check with flake8
flake8 pettingllms/

# Type checking with mypy
mypy pettingllms/
```

## Areas for Contribution

### 1. New Environments

Add support for new tasks:

- Implement environment class
- Add dataset loader
- Create configuration
- Add documentation
- Write tests

### 2. Algorithm Improvements

Enhance AT-GRPO or add new algorithms:

- Implement in `pettingllms/trainer/`
- Add tests and benchmarks
- Document changes
- Update training scripts

### 3. Performance Optimization

Improve training efficiency:

- Profile bottlenecks
- Optimize critical paths
- Add caching/memoization
- Parallelize where possible

### 4. Documentation

Improve documentation:

- Fix typos and errors
- Add examples
- Create tutorials
- Improve API docs

### 5. Testing

Expand test coverage:

- Unit tests
- Integration tests
- Performance tests
- Regression tests

## Coding Guidelines

### Code Organization

```python
# Good: Clear structure
class MyEnvironment:
    def __init__(self, config):
        self.config = config
        self._setup()
    
    def reset(self):
        """Reset environment to initial state."""
        pass
    
    def step(self, action):
        """Execute action and return observation."""
        pass
```

### Documentation

```python
def train_agent(config, num_iterations):
    """Train agent with AT-GRPO.
    
    Args:
        config: Training configuration dict
        num_iterations: Number of training iterations
    
    Returns:
        Trained model and training statistics
    
    Example:
        >>> config = load_config("code_single_policy")
        >>> model, stats = train_agent(config, 2000)
    """
    pass
```

### Testing

```python
import pytest

def test_environment_reset():
    """Test environment reset functionality."""
    env = MyEnvironment(config)
    obs = env.reset()
    assert obs is not None
    assert env.done == False

def test_environment_step():
    """Test environment step function."""
    env = MyEnvironment(config)
    env.reset()
    obs, reward, done, info = env.step(action)
    assert reward is not None
```

## Review Process

1. **Automated Checks**
   - Linting (flake8)
   - Tests (pytest)
   - Type checking (mypy)

2. **Code Review**
   - Maintainer reviews code
   - Feedback and suggestions
   - Iterate as needed

3. **Merge**
   - Approved PR gets merged
   - Changes released in next version

## Community Guidelines

- Be respectful and inclusive
- Help others learn
- Give constructive feedback
- Acknowledge contributions

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and general discussion
- **Email**: For private inquiries

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Recognition

Contributors are recognized in:
- README.md
- Release notes
- Documentation

Thank you for contributing to PettingLLMs! 🚀

