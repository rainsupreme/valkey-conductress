"""Unified entry point for Conductress."""

import argparse
import logging
import sys

from src.config import CONDUCTRESS_LOG


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="conductress", description="Valkey Conductress"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("tui", help="Launch the TUI")
    run_parser = subparsers.add_parser("run", help="Start the task runner worker")
    run_parser.add_argument(
        "--sweep",
        action="store_true",
        help="Enable sweep mode: auto-generate historical benchmark tasks when queue is empty",
    )
    run_parser.add_argument(
        "--memory-sweep",
        action="store_true",
        help="Enable memory sweep: track per-item memory overhead across history",
    )
    run_parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Path to valkey git repo for sweep (default: ~/valkey)",
    )
    subparsers.add_parser("setup", help="Run setup/bootstrap")
    subparsers.add_parser("queue", help="Manage the task queue (list, add, remove)")
    subparsers.add_parser("compare", help="Run analysis/comparison")
    subparsers.add_parser("status", help="Show runner and task status (non-blocking)")
    sweep_parser = subparsers.add_parser(
        "sweep", help="Sweep management (export, status)"
    )
    sweep_sub = sweep_parser.add_subparsers(dest="sweep_command")
    export_parser = sweep_sub.add_parser(
        "export", help="Export sweep results to dashboard JSON"
    )
    export_parser.add_argument(
        "--platform",
        required=True,
        help="Platform identifier (e.g. amd64, arm64, intel)",
    )
    export_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: ./series-{platform}.json)",
    )
    export_parser.add_argument(
        "--push",
        type=str,
        default=None,
        help="Git repo path to commit+push the exported file to",
    )
    sweep_sub.add_parser("status", help="Show sweep progress summary")

    args, remaining = parser.parse_known_args()

    # Configure logging for all subcommands
    logging.basicConfig(
        filename=str(CONDUCTRESS_LOG),
        encoding="utf-8",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("asyncssh").setLevel(logging.WARNING)

    if args.command is None:
        parser.print_usage()
        sys.exit(0)

    if args.command == "tui":
        from src.tui import BenchmarkApp

        app = BenchmarkApp()
        app.run()

    elif args.command == "run":
        import asyncio
        import json
        import traceback
        from datetime import datetime
        from pathlib import Path

        from src.config import PROJECT_ROOT
        from src.task_runner import TaskRunner

        crash_file = PROJECT_ROOT / "last_crash.json"
        repo_path = Path(args.repo) if args.repo else None
        runner = TaskRunner(
            sweep=args.sweep, memory_sweep=args.memory_sweep, repo_path=repo_path
        )
        if args.sweep:
            print("Sweep mode enabled — will auto-generate tasks when queue is empty")
        try:
            asyncio.run(runner.run())
        except KeyboardInterrupt:
            print("Runner stopped by user.")
        except Exception:
            tb = traceback.format_exc()
            timestamp = datetime.utcnow().isoformat() + "Z"
            task_desc = str(runner.task) if runner.task else None

            # Log to main log file
            logger = logging.getLogger("conductress.crash")
            logger.critical("Runner crashed!\n%s", tb)

            # Write crash file for status command
            crash_info = {
                "timestamp": timestamp,
                "traceback": tb,
                "task": task_desc,
            }
            crash_file.write_text(json.dumps(crash_info, indent=2))

            # Also print to stderr for nohup captures
            print(f"[{timestamp}] RUNNER CRASHED:", file=sys.stderr)
            print(tb, file=sys.stderr)
            sys.exit(1)

    elif args.command == "setup":
        import asyncio

        from src import config
        from src.bootstrap import (
            SERVERS,
            ensure_server_ssh_fingerprints,
            ensure_ssh_key,
            update_host_list,
        )

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        logger.info("⊹˚₊‧───Starting update/setup───‧₊˚⊹")

        ensure_ssh_key()
        asyncio.run(ensure_server_ssh_fingerprints())

        update_servers = SERVERS.copy()
        if config.ServerInfo("localhost", "", "localhost") not in update_servers:
            update_servers.append(config.ServerInfo("localhost", "", "localhost"))

        asyncio.run(update_host_list(update_servers))
        logger.info("Update/setup complete!")

    elif args.command == "queue":
        from src.cli import main as cli_main

        sys.exit(cli_main(["queue"] + remaining))

    elif args.command == "compare":
        from src.analysis import main as analysis_main

        sys.exit(analysis_main(remaining))

    elif args.command == "status":
        from src.status import print_status

        sys.exit(print_status())

    elif args.command == "sweep":
        from pathlib import Path

        from src.sweep.coordinator import SWEEP_STATE_FILE
        from src.sweep.exporter import export_series
        from src.sweep.planner import SweepState

        if args.sweep_command == "export":
            state = SweepState.load(SWEEP_STATE_FILE)
            points_count = sum(1 for p in state.points.values() if p.value is not None)
            if points_count == 0:
                print("No sweep results to export yet.")
                sys.exit(1)

            platform = args.platform
            output = (
                Path(args.output) if args.output else Path(f"series-{platform}.json")
            )
            platform_labels = {
                "amd64": "amd64/epyc-9r14/zen4",
                "arm64": "arm64/c7g.metal/graviton3",
                "intel": "intel/sapphire-rapids/8488c",
            }
            platform_str = platform_labels.get(platform, platform)

            export_series(state, output, platform=platform_str)
            print(f"Exported {points_count} points to {output}")

            if args.push:
                import shutil
                import subprocess

                repo_path = Path(args.push)
                dest = repo_path / "data" / output.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output, dest)
                subprocess.run(
                    ["git", "-C", str(repo_path), "add", str(dest)], check=True
                )
                result = subprocess.run(
                    ["git", "-C", str(repo_path), "diff", "--cached", "--quiet"],
                    capture_output=True,
                )
                if result.returncode != 0:  # there are staged changes
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(repo_path),
                            "commit",
                            "-m",
                            f"Update {output.name}: {points_count} points",
                        ],
                        check=True,
                    )
                    subprocess.run(
                        ["git", "-C", str(repo_path), "push"],
                        check=True,
                    )
                    print(f"Pushed to {repo_path}")
                else:
                    print("No changes to push (data unchanged)")

        elif args.sweep_command == "status":
            state = SweepState.load(SWEEP_STATE_FILE)
            from src.sweep.planner import SweepPlanner

            planner = SweepPlanner(state)
            completed = sum(1 for p in state.points.values() if p.value is not None)
            failed = sum(
                1 for p in state.points.values() if p.status.name == "BUILD_FAILED"
            )
            segments = planner.get_unresolved_segments()
            print(f"Commits tracked: {len(state.merge_commits)}")
            print(f"Points completed: {completed}")
            print(f"Build failures: {failed}")
            print(f"Landmarks: {len(state.landmarks)}")
            print(f"Unresolved segments (>{state.threshold*100:.0f}%): {len(segments)}")
            if segments:
                top = segments[0]
                print(
                    f"Largest gap: {top.abs_delta*100:.1f}% ({top.commit_count} commits)"
                )

        else:
            sweep_parser.print_usage()
            sys.exit(1)


if __name__ == "__main__":
    main()
