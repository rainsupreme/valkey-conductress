== Purpose

The intention of this project is to queue and run various types of benchmarks on Valkey. It is written with the assumption that there is a separate machine(s) to run valkey-server, which are different from the machine conducting the tests and generating load, which is always localhost. I intend to make this more flexible in the future.

Currently it only uses one server host, except for replication tests which use 2+. I intend to make task allocation more efficient in the future as well, time permitting.

This is unsanctioned work. The intention is to migrate my effort to the Valkey project's official test framework, which is integrated with github actions.

== Quick Start

1. Install git and python 3
2. git clone this repo to ~/conductress
3. Take a look at src/config.py and edit as desired. This may invalidate the specifics of some steps below.
4. List your server machines in servers.json, following the example of servers.json.example
5. Copy a ssh keyfile to access the servers to ~/conductress/server-keyfile.pem
6. run `python -m src.setup`. It may prompt you to make manual fixes, and you may need to run it more than once if it installs its own dependencies.
7. There is a TUI for monitoring status and queuing benchmark tasks: `python -m src.tui`. There is a worker that actually executes the tasks: `python -m src.task_runner`
8. ???
9. Profit

== Current Features

- Works on Redhat, Amazon Linux 2023, and Ubuntu (last I checked)
- perf throughput test based on valkey-benchmark. Collects data at 4Hz. Pins CPUs for greater performance and consistency.
- mem efficiency test. Adds a large number of items of a specified size and measures what additional overhead memory was allocated by Valkey. Can easily specify a range of value sizes
- perf tests can optionally gather flame graphs
- supports adding fork repos. This is useful for testing exploratory work or other commits that aren't merged into the main Valkey project.

== Feature plan/wishlist

- replication test is a WIP. Needs automated tests, needs better measurement. Has flame graphs though.
- generate more human-readable and accessible output reports. Also data that is easy to export to Excel, Google Sheets, etc.
- cpu hardware metrics: IPC, cache hit rates, context switches, etc.
- flexible and efficient scheduling to make use of all hardware available
- calculate meaningful statistics
- monitor all commits on a branch automatically
- compare performance/etc between two arbitrary commits
