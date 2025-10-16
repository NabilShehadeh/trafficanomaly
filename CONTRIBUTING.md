# Contributing to Network Traffic Anomaly Detection System

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Network Traffic Anomaly Detection System.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/traffic-anomaly-detection.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.txt`
5. Run tests: `python trafficanomaly.py test`

## Development Workflow

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused

### Testing
- Write tests for new features
- Ensure all tests pass: `python trafficanomaly.py test`
- Aim for good test coverage

### Documentation
- Update README.md for significant changes
- Add docstrings to new functions/classes
- Update API documentation if needed

## Submitting Changes

1. Make your changes
2. Run tests: `python trafficanomaly.py test`
3. Run linting: `flake8 src/` and `black src/`
4. Commit your changes: `git commit -m "Add feature: description"`
5. Push to your fork: `git push origin feature/your-feature-name`
6. Create a Pull Request

## Pull Request Guidelines

- Provide a clear description of changes
- Reference any related issues
- Ensure CI/CD pipeline passes
- Request review from maintainers

## Issue Reporting

When reporting issues, please include:
- Operating system and Python version
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant error messages/logs

## Feature Requests

For feature requests, please:
- Describe the feature clearly
- Explain the use case
- Consider implementation complexity
- Check for existing similar requests

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's coding standards

Thank you for contributing! 🎉
