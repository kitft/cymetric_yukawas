# CLAUDE.md - CYMetric Codebase Guide

## Build/Test Commands
- Install: `pip install -e .`
- Run tests: `python -m pytest tests/`
- Run single test: `python -m pytest tests/test_file.py::TestClass::test_function`
- Test order: Run `test_pointgen.py` before `test_tfmodels.py`
- Generate documentation: `cd docs && make html`

## Style Guidelines
- **Classes**: CamelCase (PhiFSModel)
- **Functions/Variables**: snake_case
- **Private methods**: prefixed with underscore
- **Constants**: ALL_CAPS
- **Imports**: Group standard, third-party, then local imports
- **Docstrings**: reStructuredText format with LaTeX for math formulas
- **Error handling**: Use assertions and parameter validation
- **Type hints**: Document parameter types in docstrings
- **Optimization**: Use numpy/tensorflow vectorized operations
- **Logging**: Use Python's logging module with consistent module naming

## Mathematical Conventions
For detailed mathematical conventions and normalizations, see `./assets/conventions.pdf`