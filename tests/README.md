# Test Organization

## Structure

```
tests/
├── unit/          # Fast, isolated tests with mocked dependencies
├── integration/   # Slower tests with real components
└── conftest.py    # Shared fixtures
```

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (fast, < 1 second)
pytest tests/unit

# Run only integration tests (slower, requires valkey)
pytest tests/integration

# Run specific test file
pytest tests/unit/test_file_protocol.py
```
