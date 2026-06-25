# Python Coding Style

## Standards
- Follow **PEP 8** conventions
- Use **type annotations** on all function signatures

## Immutability
Prefer immutable data structures (frozen dataclasses, NamedTuple).

## Formatting
Use **black**, **isort**, **ruff** for formatting and linting.

## Environment Variables
```python
import os
api_key = os.environ["KEY"]  # Raises KeyError if missing
```
