# Contributing to Neural Wavefront

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/neural_wavefront.git
   cd neural_wavefront
   ```

2. **Install Dependencies**
   ```bash
   uv pip install -e ".[dev]"
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run before committing:
```bash
# Format code
black src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Check linting
flake8 src/ scripts/ tests/

# Type checking
mypy src/
```

### Testing

All new features must include tests:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Documentation

- All public functions must have docstrings (NumPy or Google style)
- Update README.md if adding major features
- Update AGENT.md if changing architecture or physics implementation

## Coding Standards

### Required

1. **Type Hints**: All function signatures must include type hints
   ```python
   def compute_psf(aperture: np.ndarray, phase: np.ndarray) -> np.ndarray:
       """Compute PSF from aperture and phase."""
       ...
   ```

2. **Docstrings**: Use NumPy-style docstrings
   ```python
   def my_function(param1: int, param2: str) -> bool:
       """
       Brief description.

       Longer description if needed.

       Parameters
       ----------
       param1 : int
           Description of param1
       param2 : str
           Description of param2

       Returns
       -------
       bool
           Description of return value
       """
   ```

3. **Configuration**: No hardcoded values—use `configs/config.yaml`

4. **Imports**: Organize as standard library, third-party, local
   ```python
   # Standard library
   from pathlib import Path
   import logging

   # Third-party
   import numpy as np
   import torch

   # Local
   from src.utils.config import load_config
   ```

### Prohibited

- ❌ Hardcoded numerical constants (except π, 2π, etc.)
- ❌ `print()` statements (use `logging` module)
- ❌ Relative imports that break from different directories
- ❌ Package managers other than `uv`

## Pull Request Process

1. **Update Tests**: Ensure all tests pass
2. **Update Documentation**: Add/update docstrings and README if needed
3. **Code Quality**: Run formatting and linting tools
4. **Commit Message**: Use clear, descriptive commit messages
   ```
   feat: Add Zernike polynomial generator
   fix: Correct FFT normalization in PSF computation
   docs: Update README with installation instructions
   ```

5. **Create PR**: Submit pull request with description of changes

## Project Structure Guidelines

When adding new modules, follow this structure:

```
src/
├── optics/         # Physics-related code
├── data/           # Data generation and loading
├── models/         # Neural network architectures
├── training/       # Training loops and metrics
└── utils/          # General utilities
```

## Questions?

- Open an issue for bugs or feature requests
- Check [AGENT.md](AGENT.md) for detailed implementation guidelines
- Contact: [your.email@example.com](mailto:your.email@example.com)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
