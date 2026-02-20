# Technology Stack

## Programming Languages
- **Python 3** - Primary language for all application code
- **C** - Valkey server source code (embedded in `/valkey/`)

## Core Dependencies
See `requirements/pip-requirements.txt` for runtime dependencies and `requirements/pip-requirements-dev.txt` for development dependencies.

## Build Systems
- **Make** - Valkey compilation
- **pip** - Python package management

## Development Commands

### Running the Application
```bash
# Start the TUI for monitoring and task creation
python -m src.tui

# Start the task runner worker
python -m src.task_runner

# Initial setup (install dependencies, configure servers)
python -m src.setup
```

### Testing
```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run all tests
pytest tests/
```

## Configuration Files
- `servers.json` - Remote server definitions (optional, falls back to `servers.default.json`)
- `server-keyfile.pem` - SSH private key for server access
- `src/config.py` - Application configuration constants

## Key Configuration Parameters
- `PERF_BENCH_KEYSPACE = 3_000_000` - Keys for performance tests
- `PERF_BENCH_CLIENTS = 1200` - Concurrent clients
- `PERF_BENCH_THREADS = 64` - Benchmark threads
- `MEM_TEST_ITEM_COUNT = 5_000_000` - Items for memory tests
- `TUI_REFRESH_INTERVAL = 15` - TUI refresh rate (seconds)

## Remote Server Requirements
- SSH access with key-based authentication
- Valkey build dependencies (gcc, make, etc.)
- Sufficient memory for benchmark workloads
- Linux OS (RedHat, Amazon Linux 2023, or Ubuntu)

## Result Formats
- **JSONL** - Line-delimited JSON for benchmark data
- **SVG** - Flame graphs
