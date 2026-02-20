# Project Structure

## Directory Organization

### `/src/` - Core Application Code
- `tui.py` - Terminal UI using Textual framework for monitoring and task creation
- `task_runner.py` - Worker process that executes queued benchmark tasks
- `task_queue.py` - Task queue management and persistence
- `tui_data_service.py` - Data service layer for TUI
- `config.py` - Central configuration (servers, paths, benchmark parameters)
- `server.py` - Remote server management and SSH operations
- `replication_group.py` - Manages groups of Valkey instances for replication tests
- `file_protocol.py` - Protocol for reading benchmark result files
- `base_task_visualizer.py` - Base class for task visualization widgets
- `bootstrap.py` - Project setup and dependency installation
- `utility.py` - Shared utility functions

### `/src/tasks/` - Task Implementations
- `task_perf_benchmark.py` - Performance/throughput benchmarking
- `task_mem_efficiency.py` - Memory overhead measurement
- `task_full_sync.py` - Replication synchronization testing

### `/tests/` - Test Suite
- `unit/` - Unit tests for individual components (run with `pytest tests/unit`)
- `integration/` - Integration tests for end-to-end workflows
- `conftest.py` - Pytest configuration and fixtures

### `/benchmark_queue/` - Task Queue Storage
JSON files representing queued benchmark tasks

### `/results/` - Benchmark Results
JSONL files with timestamped benchmark data, analysis reports, flame graphs
SVG files for flame graphs
and any other results, analysis, or output

### `/valkey/` - Valkey Source Code
Embedded Valkey repository for building needed binaries: valkey-cli and valkey-benchmark

### `/requirements/` - Dependencies
- `pip-requirements.txt` - Python dependencies
- `pip-requirements-dev.txt` - Development dependencies
- Platform-specific requirements (Ubuntu, RHEL, Amazon Linux)

## Core Components

### Task Queue System
- Tasks defined as JSON files in `benchmark_queue/`
- `task_runner.py` polls queue and executes tasks
- Task types: perf-{command}, mem-{command}, sync-{mode}

### Remote Server Management
- Server definitions in `servers.json` (falls back to `servers.default.json`)
- SSH access via `server-keyfile.pem`
- Automated Valkey compilation and deployment
- Multi-instance support on single host (ports 9000+)

### TUI Architecture
- Tabbed interface: Status, Queue, Create Task
- Real-time task monitoring with visualizers
- DataTable widgets for task lists
- Form-based task creation (Perf, Mem, Sync)

## Architectural Patterns
- **Separation of Concerns**: TUI, task execution, and data services are decoupled
- **Worker Pattern**: Background workers for data fetching and task execution
- **Protocol Pattern**: File protocol abstraction for reading benchmark results
- **Factory Pattern**: Task visualizers created based on task type
- **Configuration Centralization**: All settings in `config.py`
