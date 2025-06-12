"""Conductress TUI application for Valkey benchmarking tasks."""

import logging
from datetime import datetime
from itertools import product
from typing import Callable, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.validation import ValidationResult, Validator
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Pretty,
    SelectionList,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.widgets.selection_list import Selection

from . import config
from .task_perf_benchmark import TestPerf
from .task_queue import Task, TaskQueue

logger = logging.getLogger(__name__)


class BenchmarkApp(App):
    """Main application class for the benchmark app."""

    DARK = True

    CSS = """
    .form-label {
        margin-top: 1;
        margin-bottom: 0;
    }
    .form-input {
        margin-top: 0;
        margin-bottom: 1;
    }
    .switch-label {
        height: 3;
        content-align: center middle;
        width: auto;
    }
    .switch-container {
        padding: 1 0;
        height: 5;
    }
    #test-list {
        padding: 0;
        margin: 1 0;
        border: solid $border;
        height: auto;
        width: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-create-task", id="root-tabs"):
            with TabPane("Status", id="tab-status"):
                yield Static("TODO: implement view of current tasks")
            with TabPane("Queue", id="tab-queue"):
                yield Static("Last update: Never", id="queue-table-status")
                yield DataTable(id="queue-table", cursor_type="row")
            with TabPane("Create Task", id="tab-create-task"):
                with TabbedContent():
                    with TabPane("Perf"):
                        yield PerfTaskForm(self.refresh_data)
                    with TabPane("Mem"):
                        yield MemTaskForm(self.refresh_data)
                    with TabPane("Sync"):
                        yield SyncTaskForm(self.refresh_data)
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.theme = "catppuccin-mocha"
        self.set_interval(10, self.refresh_data)
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh the table data."""
        queue = TaskQueue()
        tasks = queue.get_all_tasks()

        tabs = self.query_one("#root-tabs", TabbedContent)
        table = self.query_one("#queue-table", DataTable)
        status = self.query_one("#queue-table-status", Static)

        tabs.get_tab("tab-queue").label = f"Queue ({len(tasks)})"

        table.clear()

        if not table.columns:
            table.add_columns(
                "Timestamp",
                "Type",
                "Test",
                "Source:Specifier",
                "Threads",
                "Pipeline",
                "ValSize",
                "Expire",
                "Profiling",
            )

        for task in tasks:
            table.add_row(
                task.timestamp,
                task.task_type,
                task.test,
                f"{task.source}:{task.specifier}",
                str(task.io_threads),
                str(task.pipelining),
                str(task.val_size),
                str(task.has_expire),
                str(task.profiling_sample_rate > 0),
            )

        # Update the status message
        status.update(f"Last update: {datetime.now().strftime('%H:%M')}")


class SourceSpeciferValidator(Validator):
    """Validator for source:specifier list input"""

    def validate(self, value: str) -> ValidationResult:
        """Validate the source:specifier list input"""
        _, error = self.parse_source_specifier_list(value)
        if error:
            return self.failure(error)
        return self.success()

    @staticmethod
    def parse_source_specifier_list(input_str: str) -> tuple[list[tuple[str, str]], Optional[str]]:
        """Parse the source:specifier list into two separate lists"""
        if not input_str:
            return [], "list cannot be empty"

        specifiers: list[tuple[str, str]] = []
        for item in input_str.split(","):
            if item.count(":") != 1:
                return [], f"Invalid item format: {item}"
            source, specifier = item.split(":")
            source = source.strip()
            specifier = specifier.strip()
            if source not in config.REPO_NAMES and source != config.MANUALLY_UPLOADED:
                return [], f"Invalid source: {source}"
            if not specifier:
                return [], f"Invalid specifier: {specifier}"
            specifiers.append((source, specifier))
        return specifiers, None


class CommaSeparatedIntsValidator(Validator):
    """Validator for comma-separated positive integers"""

    def validate(self, value: str) -> ValidationResult:
        """Validate the comma-separated numbers input"""
        if not value:
            return self.failure("Input cannot be empty")
        try:
            numbers = [int(x) for x in value.split(",")]
            if any(x <= 0 for x in numbers):
                return self.failure("All numbers must be positive")
        except ValueError:
            return self.failure("Invalid input format")
        return self.success()


class BaseTaskForm(ScrollableContainer):

    def __init__(self, update_queue_fn: Callable[[], None], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_queue_view = update_queue_fn
        self.queue = TaskQueue()

    def queue_tasks(self, tasks: list[Task]) -> None:
        """Queue the tasks and update the view"""
        for task in tasks:
            self.queue.submit_task(task)
        self.update_queue_view()
        self.notify(f"{len(tasks)} tasks queued 🌧 ♥")


class PerfTaskForm(BaseTaskForm):
    """Form for creating a performance test task"""

    def compose(self) -> ComposeResult:
        yield Label(
            f"Source:Specifier list (comma-separated)\nAvailable sources: {', '.join(config.REPO_NAMES)}",
            classes="form-label",
        )
        yield Input(
            placeholder=(
                f"{config.REPO_NAMES[0]}:unstable, "
                f"{config.REPO_NAMES[-1]}:sha1-hash, "
                f"{config.MANUALLY_UPLOADED}:local-path"
            ),
            id="specifiers",
            classes="form-input",
            validators=[SourceSpeciferValidator()],
        )

        yield Label("Pipelining (comma-separated)", classes="form-label")
        yield Input(
            value="4",
            placeholder="1, 4, 8",
            id="pipelining",
            classes="form-input",
            validators=[CommaSeparatedIntsValidator()],
        )

        yield Label("IO Threads (comma-separated)", classes="form-label")
        yield Input(
            value="9",
            placeholder="1, 9",
            id="io-threads",
            classes="form-input",
            validators=[CommaSeparatedIntsValidator()],
        )

        yield Label("Sizes (comma-separated)", classes="form-label")
        yield Input(
            value="512",
            placeholder="256, 512",
            id="sizes",
            classes="form-input",
            validators=[CommaSeparatedIntsValidator()],
        )

        yield Horizontal(
            Switch(animate=False, value=True, id="preload-keys"),
            Static("Preload Keys", classes="switch-label"),
            Switch(animate=False, value=False, id="expire-keys"),
            Static("Expire Keys", classes="switch-label"),
            Switch(animate=False, value=False, id="profiling"),
            Static("Profiling", classes="switch-label"),
            classes="switch-container",
        )

        tests = tuple(Selection[str](name, name) for name in TestPerf.tests)
        yield SelectionList[str](
            *tests,
            id="test-list",
        )

        yield Button("Submit", variant="primary", id="submit-perf-task")

        yield Pretty("", id="validation-errors")

    def on_mount(self) -> None:
        """Called when the form is mounted"""
        self.query_one("#test-list", SelectionList).border_title = "Tests"

    @on(Input.Changed)
    def show_validation(self, event: Input.Changed) -> None:
        """Show validation errors"""
        if not event.validation_result:
            return
        if event.validation_result.is_valid:
            self.query_one("#validation-errors", Pretty).update("")
        else:
            self.query_one("#validation-errors", Pretty).update(event.validation_result.failure_descriptions)

    @on(Button.Pressed, "#submit-perf-task")
    def submit_task(self) -> None:
        """Submit the task to the queue"""
        try:
            source_specifier_list = self.query_one("#specifiers", Input).value
            pipelining: list[int] = [int(x) for x in self.query_one("#pipelining", Input).value.split(",")]
            io_threads: list[int] = [int(x) for x in self.query_one("#io-threads", Input).value.split(",")]
            sizes: list[int] = [int(x) for x in self.query_one("#sizes", Input).value.split(",")]
            tests: list[str] = self.query_one("#test-list", SelectionList).selected
            preload_keys: bool = self.query_one("#preload-keys", Switch).value
            expire_keys: bool = self.query_one("#expire-keys", Switch).value
            profiling: bool = self.query_one("#profiling", Switch).value
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")
            return

        if not tests:
            self.notify("No tests selected", severity="error")
            return

        specifiers, error = SourceSpeciferValidator.parse_source_specifier_list(source_specifier_list)
        if error:
            self.notify(f"source:specifier list: {error}", severity="error")
            return

        all_tests = list(
            product(
                sizes,
                pipelining,
                io_threads,
                tests,
                specifiers,
            )
        )
        profiling_sample_rate = 399 if profiling else -1

        tasks = []
        for size, pipe, thread, test, specifier in all_tests:
            task = Task.perf_task(
                test=test,
                source=specifier[0],
                specifier=specifier[1],
                val_size=size,
                io_threads=thread,
                pipelining=pipe,
                warmup=5,
                duration=60,
                profiling_sample_rate=profiling_sample_rate,
                has_expire=expire_keys,
                preload_keys=preload_keys,
                replicas=-1,
            )
            tasks.append(task)
        self.queue_tasks(tasks)


class MemTaskForm(BaseTaskForm):
    """Form for creating a performance test task"""

    def compose(self) -> ComposeResult:
        yield Label(
            f"Source:Specifier list (comma-separated)\nAvailable sources: {', '.join(config.REPO_NAMES)}",
            classes="form-label",
        )
        yield Input(
            placeholder=(
                f"{config.REPO_NAMES[0]}:unstable, "
                f"{config.REPO_NAMES[-1]}:sha1-hash, "
                f"{config.MANUALLY_UPLOADED}:local-path"
            ),
            id="specifiers",
            classes="form-input",
            validators=[SourceSpeciferValidator()],
        )

        yield Label("Sizes (comma-separated)", classes="form-label")
        yield Input(
            value="512",
            placeholder="256, 512",
            id="sizes",
            classes="form-input",
            validators=[CommaSeparatedIntsValidator()],
        )

        yield Horizontal(
            Switch(animate=False, value=False, id="expire-keys"),
            Static("Expire Keys", classes="switch-label"),
            classes="switch-container",
        )

        tests = tuple(
            Selection[str](name, name) for name in TestPerf.tests
        )  # TODO shouldn't this be TestMem or something?
        yield SelectionList[str](
            *tests,
            id="test-list",
        )

        yield Button("Submit", variant="primary", id="submit-mem-task")

        yield Pretty("", id="validation-errors")

    def on_mount(self) -> None:
        """Called when the form is mounted"""
        self.query_one("#test-list", SelectionList).border_title = "Tests"

    @on(Input.Changed)
    def show_validation(self, event: Input.Changed) -> None:
        """Show validation errors"""
        if not event.validation_result:
            return
        if event.validation_result.is_valid:
            self.query_one("#validation-errors", Pretty).update("")
        else:
            self.query_one("#validation-errors", Pretty).update(event.validation_result.failure_descriptions)

    @on(Button.Pressed, "#submit-mem-task")
    def submit_task(self) -> None:
        """Submit the task to the queue"""
        try:
            source_specifier_list = self.query_one("#specifiers", Input).value
            sizes: list[int] = [int(x) for x in self.query_one("#sizes", Input).value.split(",")]
            tests: list[str] = self.query_one("#test-list", SelectionList).selected
            expire_keys: bool = self.query_one("#expire-keys", Switch).value
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")
            return

        if not tests:
            self.notify("No tests selected", severity="error")
            return

        specifiers, error = SourceSpeciferValidator.parse_source_specifier_list(source_specifier_list)
        if error:
            self.notify(f"source:specifier list: {error}", severity="error")
            return

        all_tests = list(
            product(
                sizes,
                tests,
                specifiers,
            )
        )

        tasks = []
        for size, test, specifier in all_tests:
            task = Task.mem_task(
                source=specifier[0], specifier=specifier[1], val_size=size, test=test, has_expire=expire_keys
            )
            tasks.append(task)
        self.queue_tasks(tasks)


class SyncTaskForm(BaseTaskForm):
    """Form for creating a full sync benchmark task"""

    def compose(self) -> ComposeResult:
        yield Static("TODO: implement sync task form")


if __name__ == "__main__":
    logging.basicConfig(filename=config.CONDUCTRESS_LOG, encoding="utf-8", level=logging.DEBUG)
    app = BenchmarkApp()
    app.run()
