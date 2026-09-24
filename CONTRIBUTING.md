# Contributing to CUDA Neural Network Engine

Thank you for your interest in contributing. This document outlines the process for contributing to this project.

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- CUDA version and GPU model
- Error messages or logs

Use the [bug report template](https://github.com/chinmayasameeru/cuda-neural-network-engine/issues/new?template=bug_report.md) when creating a new issue.

## Feature Requests

Feature requests are welcome. Please use the [feature request template](https://github.com/chinmayasameeru/cuda-neural-network-engine/issues/new?template=feature_request.md) and include:

- A clear description of the proposed feature
- Why this feature would be useful
- Any relevant technical considerations

## Pull Request Process

1. Fork the repository and create a branch from `main`
2. Implement your changes
3. Add or update tests as appropriate
4. Ensure all existing tests pass: `python tests/test_engine.py`
5. Update documentation if your changes affect public APIs
6. Submit a pull request using the [PR template](https://github.com/chinmayasameeru/cuda-neural-network-engine/blob/main/.github/PULL_REQUEST_TEMPLATE.md)

## Code Standards

### C++ Code

- Follow the existing code style (4-space indentation, snake_case naming)
- Use `const` where appropriate
- Prefer `std::vector` over raw arrays
- Document public APIs with comments
- Avoid exceptions in CUDA kernel code

### Python Code

- Follow PEP 8 style guidelines
- Use type hints for public functions
- Write docstrings for modules and public classes

### CUDA Code

- Use `float` (FP32) as the default precision
- Minimize host-device memory transfers
- Use shared memory where it provides benefit
- Document kernel launch configurations

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.
