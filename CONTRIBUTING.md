# Contributing to CUDA Neural Network Engine

Thank you for considering contributing to this document. Your contributions help make this project better for everyone.

## Ways to Contribute

There are many ways to contribute to this project:

- **Code**: Fix bugs, implement new features, improve performance
- **Documentation**: Fix typos, clarify explanations, add examples
- **Testing**: Add test cases, improve test coverage, find edge cases
- **Feedback**: Report bugs, suggest features, share your experience

## Getting Started

### Fork and Clone

```bash
git clone https://github.com/chinmayasameeru/cuda-neural-network-engine.git
cd cuda-neural-network-engine
```

### Set Up Development Environment

```bash
pip install pybind11 numpy pytest black flake8
```

### Build the Project

```bash
python setup.py build_ext --inplace
```

### Run Tests

```bash
python tests/test_engine.py
# or
make test
```

## Reporting Issues

When reporting an issue, please include:

1. **Description**: A clear summary of the problem
2. **Reproduction**: Minimal code to reproduce the issue
3. **Expected**: What you expected to happen
4. **Actual**: What actually happened
5. **Environment**:
   - Operating system
   - CUDA version
   - GPU model and memory
   - Python version
   - Compiler version

Please use the [bug report template](https://github.com/chinmayasameeru/cuda-neural-network-engine/issues/new?template=bug_report.md) when creating issues.

## Feature Requests

Feature requests are welcome. When submitting one, please explain:

- What the feature would do
- Why it would be useful
- How it might be implemented (if you have ideas)

## Pull Request Process

1. Create a branch from `main`
2. Make your changes
3. Add tests if applicable
4. Run the test suite: `make test`
5. Run the linter: `make lint`
6. Update documentation if needed
7. Submit a pull request

## Code Style

### C++ Code

- Indentation: 4 spaces (no tabs)
- Naming: `snake_case` for variables and functions, `CamelCase` for classes
- Braces: Opening brace on the same line
- Comments: Document public APIs

Example:

```cpp
class Layer {
public:
    virtual ~Layer() = default;
    virtual TensorPtr forward(TensorPtr input) = 0;
    virtual std::string name() const = 0;

protected:
    bool training_ = true;
};
```

### Python Code

- Follow PEP 8
- Line length: maximum 100 characters
- Use type hints for function signatures
- Write docstrings for public functions

### CUDA Code

- Use `float` (FP32) as default precision
- Minimize memory transfers between host and device
- Use shared memory when beneficial
- Comment kernel launch configurations

## Testing

All new features should include tests. Tests are in `tests/` and use the standard `assert` pattern.

```python
def test_new_feature():
    model = cnn.Sequential()
    model.add("Linear", [10, 5])
    X = np.random.randn(2, 10).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (2, 5)
```

## Documentation

If your changes affect the public API or add new features, please update:

- `README.md` if applicable
- Code comments
- Any relevant documentation in `docs/`

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.
