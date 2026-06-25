# Testing Requirements

## Minimum Coverage: 80%

## Test Types Required
1. **Unit Tests** - Individual functions in isolation
2. **Integration Tests** - API endpoints, database operations
3. **E2E Tests** - Critical user flows

## TDD Workflow (Mandatory)
1. Write test first (RED) - it must FAIL
2. Write minimal implementation (GREEN) - it must PASS
3. Refactor (IMPROVE) - tests stay green

## Framework
- Use **pytest** with `pytest.mark` for categorization

## Edge Cases to Test
- Null/undefined input, empty arrays/strings, invalid types
- Boundary values (min/max), error paths
- Race conditions, large data (10k+ items)
- Special characters (Unicode, SQL chars)

## Coverage Check
```bash
pytest --cov=src --cov-report=term-missing
```

## Test Anti-Patterns
- Testing implementation details instead of behavior
- Tests depending on each other (shared state)
- Asserting too little (passing tests that don't verify)
- Not mocking external dependencies
