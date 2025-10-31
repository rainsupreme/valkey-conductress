# Product Overview

## Purpose
Conductress is a benchmark orchestration framework for Valkey (Redis fork). It queues and executes various benchmark types on remote Valkey server instances while generating load from localhost. The project aims to provide comprehensive performance, memory efficiency, and replication testing capabilities.

## Key Features
- **Performance Benchmarking**: Throughput tests using valkey-benchmark with 4Hz data collection, CPU pinning for consistency
- **Memory Efficiency Testing**: Measures memory overhead by adding millions of items with specified sizes
- **Replication Testing**: Tests synchronization between multiple Valkey instances (WIP)
- **Flame Graph Support**: Optional flame graph generation during performance tests
- **Multi-Repository Support**: Test different Valkey forks and branches simultaneously
- **Task Queue System**: Queue and manage multiple benchmark tasks with priority handling
- **TUI Interface**: Real-time monitoring and task management via terminal UI
- **Cross-Platform**: Supports RedHat, Amazon Linux 2023, and Ubuntu

## Target Users
- Valkey developers testing performance improvements
- Infrastructure engineers evaluating Valkey deployments
- Contributors benchmarking experimental features
- Teams comparing performance across commits/branches

## Use Cases
- Continuous performance regression testing
- Memory overhead analysis for different data patterns
- Replication lag and throughput measurement
- CPU tuning and optimization validation
- Comparing performance between Valkey versions or forks
