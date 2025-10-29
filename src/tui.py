"""Conductress TUI application for Valkey benchmarking tasks."""

import glob
import logging
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable, Iterator, Optional, TypeVar, Union

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
    SelectionList,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.widgets.selection_list import Selection

from src.file_protocol import FileProtocol
from src.tasks.task_full_sync import SyncTaskData
from src.tasks.task_mem_efficiency import MemTaskData, MemTaskRunner
from src.tasks.task_perf_benchmark import PerfTaskData, PerfTaskRunner

from . import config
from .task_queue import BaseTaskData, TaskQueue
from .utility import HumanByte, HumanNumber, HumanTime

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
                yield Static("Last update: Never", id="status-table-status")
                yield DataTable(id="status-table", cursor_type="row")
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
        self._refresh_queue_data()
        self._refresh_status_data()

    def _refresh_queue_data(self) -> None:
        """Refresh the queue table data."""
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
                "Source:Specifier",
                "Description",
                "Note",
            )

        for task in tasks:
            task_type = task.task_type
            if task_type.endswith("TaskData"):
                task_type = task_type[:-8]
            table.add_row(
                task.timestamp,
                task_type,
                f"{task.source}:{task.specifier}",
                task.short_description(),
                task.note,
            )

        # Update the status message
        status.update(f"Last update: {datetime.now().strftime('%H:%M')}")

    def _refresh_status_data(self) -> None:
        """Refresh the status table data."""

        # Find all active benchmark directories
        benchmark_dirs = glob.glob("/tmp/benchmark_*")

        tabs = self.query_one("#root-tabs", TabbedContent)
        table = self.query_one("#status-table", DataTable)
        status_label = self.query_one("#status-table-status", Static)

        running_tasks = []

        for dir_path in benchmark_dirs:
            try:
                # Extract task ID from directory name
                task_id = Path(dir_path).name.replace("benchmark_", "")
                protocol = FileProtocol(task_id, Path("/tmp"))

                # Read status if available
                task_status = protocol.read_status()
                if task_status:
                    # Calculate progress
                    progress = "N/A"
                    if task_status.steps_total and task_status.steps_completed is not None:
                        pct = (task_status.steps_completed / task_status.steps_total) * 100
                        progress = f"{pct:.0f}% ({task_status.steps_completed}/{task_status.steps_total})"

                    running_tasks.append(
                        {
                            "task_id": task_id,
                            "state": task_status.state,
                            "pid": task_status.pid or "N/A",
                            "progress": progress,
                        }
                    )
            except Exception:
                # Skip directories that don't have valid protocol files
                continue

        tabs.get_tab("tab-status").label = f"Status ({len(running_tasks)})"

        table.clear()

        if not table.columns:
            table.add_columns(
                "Task ID",
                "State",
                "PID",
                "Progress",
            )

        for task in running_tasks:
            table.add_row(
                task["task_id"],
                task["state"],
                str(task["pid"]),
                task["progress"],
            )

        status_label.update(f"Last update: {datetime.now().strftime('%H:%M:%S')}")


class SourceSpeciferValidator(Validator):
    """Validator for source:specifier list input"""

    def validate(self, value: str) -> ValidationResult:
        """Validate the source:specifier list input"""
        _, error = self.parse_source_specifier_list(value)
        if error:
            return self.failure(error)
        return self.success()

    @staticmethod
    def parse_source_specifier_list(
        input_str: str,
    ) -> tuple[list[tuple[str, str]], Optional[str]]:
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


HumanNumberType = TypeVar("HumanNumberType", bound=HumanNumber)


class RangeListValidator(Validator):
    """Validator for list of ranges: [1,500,1:5,1:256:8]"""

    def __init__(self, number_type: type[HumanNumberType]):
        super().__init__()
        self.number_type = number_type

    def validate(self, value: str) -> ValidationResult:
        _, error = self.parse_range_list(value)
        if error:
            return self.failure(error)
        return self.success()

    def parse_range_list(self, input_str: str) -> tuple[list[int], Optional[str]]:
        if not input_str:
            return [], "input cannot be empty"

        try:
            ranges = [
                [int(self.number_type.from_human(num)) for num in rangespec.split(":")]
                for rangespec in input_str.split(",")
            ]
        except ValueError as e:
            return [], str(e)

        result = []
        for rangespec in ranges:
            if len(rangespec) == 1:
                result += rangespec
            elif len(rangespec) == 3:
                if rangespec[2] == 0:
                    return [], "range step (start:end:step) value must not be zero"
                result.extend(range(rangespec[0], rangespec[1] + rangespec[2], rangespec[2]))
            else:
                return [], "ranges are of the format value, or start:end:step"
        return result, None


class SingleNumberValidator(Validator):
    """Validator for a single positive integer. Suffixes (K, M, etc) accepted."""

    def __init__(self, number_type: type[HumanNumberType]):
        super().__init__()
        self.number_type = number_type

    def parse_int(self, value: str) -> int:
        if not value:
            raise ValueError("Input cannot be empty")
        number = self.number_type.from_human(value)
        if number < 0:
            raise ValueError("Number must be positive")
        if not number.is_integer():
            raise ValueError("Number must be an integer")
        return int(number)

    def validate(self, value: str) -> ValidationResult:
        try:
            self.parse_int(value)
        except ValueError:
            return self.failure()
        return self.success()


class CommaSeparatedIntsValidator(Validator):
    """Validator for comma-separated positive integers. Suffixes (K, M, etc) accepted."""

    def __init__(self, number_type: type[HumanNumberType]):
        super().__init__()
        self.number_type = number_type

    def parse_ints(self, value: str) -> list[int]:
        if not value:
            raise ValueError("Input cannot be empty")
        numbers = [self.number_type.from_human(x) for x in value.split(",")]
        if any(x < 0 for x in numbers):
            raise ValueError("All numbers must be positive")
        if not all(x.is_integer() for x in numbers):
            raise ValueError("All numbers must be integers")
        return [int(x) for x in numbers]

    def validate(self, value: str) -> ValidationResult:
        """Validate the comma-separated numbers input"""
        try:
            self.parse_ints(value)
        except ValueError:
            return self.failure()
        return self.success()


class NumberField:
    def __init__(
        self,
        label,
        input_id,
        default,
        placeholder,
        number_type: type[HumanNumberType],
    ):
        self.label = label
        self.id = input_id
        self.default = default
        self.placeholder = placeholder
        self.number_type = number_type

        self.input = Input(
            value=self.default,
            placeholder=self.placeholder,
            id=self.id,
            classes="form-input",
            validators=[SingleNumberValidator(self.number_type)],
        )

    def widgets(self) -> Iterator[Union[Input, Label]]:
        yield Label(self.label, classes="form-label")
        yield self.input

    def value(self) -> int:
        return SingleNumberValidator(self.number_type).parse_int(self.input.value)


class NumberListField:
    def __init__(
        self,
        label,
        input_id,
        default,
        placeholder,
        number_type: type[HumanNumberType],
        allow_ranges: bool = False,
    ):
        self.label = label
        self.id = input_id
        self.default = default
        self.placeholder = placeholder
        self.number_type = number_type
        self.allow_ranges = allow_ranges

        self.input = Input(
            value=self.default,
            placeholder=self.placeholder,
            id=self.id,
            classes="form-input",
            validators=[CommaSeparatedIntsValidator(self.number_type)],
        )

    def widgets(self) -> Iterator[Union[Input, Label]]:
        yield Label(self.label, classes="form-label")
        yield self.input

    def values(self) -> list[int]:
        if self.allow_ranges:
            value_list, _ = RangeListValidator(self.number_type).parse_range_list(self.input.value)
            return value_list
        else:
            return CommaSeparatedIntsValidator(self.number_type).parse_ints(self.input.value)


class PipeliningField(NumberListField):
    def __init__(self):
        super().__init__("Pipelining (comma-separated)", "pipelining", "4", "1, 4, 8", HumanNumber)


class IOThreadsField(NumberListField):
    def __init__(self):
        super().__init__("IO Threads (comma-separated)", "io-threads", "9", "1, 9", HumanNumber)


class SizesField(NumberListField):
    def __init__(self):
        super().__init__(
            "Sizes (comma separated values or start:stop:step ranges)",
            "sizes",
            "0.5KB",
            "256, 1KB, 1KB:16KB:2KB",
            HumanByte,
            allow_ranges=True,
        )


class CountsField(NumberListField):
    def __init__(self):
        super().__init__("Value counts (comma-separated)", "counts", "10M", "1M, 30M", HumanNumber)


class BaseTaskForm(ScrollableContainer):

    def __init__(self, update_queue_fn: Callable[[], None], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_queue_view = update_queue_fn
        self.queue = TaskQueue()

    def _compose_source_specifier_input(self) -> Iterator[Union[Label, Input]]:
        """Yield source:specifier input widgets"""
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

    def _compose_note_input(self) -> Iterator[Union[Label, Input]]:
        """Yield note input widgets"""
        yield Label("Note (optional)", classes="form-label")
        yield Input(placeholder="Add a short note...", id="note", classes="form-input")

    def _compose_test_selection(self, tests: tuple[str, ...]) -> SelectionList:
        """Create test selection list widget"""
        selections = tuple(Selection[str](name, name) for name in tests)
        return SelectionList[str](*selections, id="test-list")

    def _compose_switch_row(self, *switches: tuple[str, str, bool]) -> Horizontal:
        """Create a horizontal row of switches. Each switch is (id, label, default)"""
        widgets = []
        for switch_id, label, default in switches:
            widgets.append(Switch(animate=False, value=default, id=switch_id))
            widgets.append(Static(label, classes="switch-label"))
        return Horizontal(*widgets, classes="switch-container")

    def _validate_and_get_common_inputs(self) -> tuple[list[tuple[str, str]], str, Optional[str]]:
        """Validate and return (specifiers, note, error_message)"""
        source_specifier_list = self.query_one("#specifiers", Input).value
        note = self.query_one("#note", Input).value.strip()
        
        specifiers, error = SourceSpeciferValidator.parse_source_specifier_list(source_specifier_list)
        if error:
            return [], "", f"source:specifier list: {error}"
        
        return specifiers, note, None

    def _validate_tests_selected(self, tests: list[str]) -> Optional[str]:
        """Return error message if no tests selected, None otherwise"""
        if not tests:
            return "No tests selected"
        return None

    def queue_tasks(self, tasks: list[BaseTaskData]) -> None:
        """Queue the tasks and update the view"""
        for task in tasks:
            self.queue.submit_task(task)
        self.update_queue_view()
        self.notify(f"{len(tasks)} tasks queued 🌧 ♥")


class PerfTaskForm(BaseTaskForm):
    """Form for creating a performance test task"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipelining = PipeliningField()
        self.io_threads = IOThreadsField()
        self.sizes = SizesField()
        self.warmup = NumberField("Warmup (seconds)", "warmup", "5m", "5m", HumanTime)
        self.duration = NumberField("Duration (seconds)", "duration", "1h", "1h", HumanTime)

    def compose(self) -> ComposeResult:
        for widget in self._compose_source_specifier_input():
            yield widget

        for field in (self.pipelining, self.io_threads, self.sizes, self.warmup, self.duration):
            for widget in field.widgets():
                yield widget

        yield self._compose_switch_row(
            ("preload-keys", "Preload Keys", True),
            ("expire-keys", "Expire Keys", False),
            ("profiling", "Profiling", False),
        )

        yield self._compose_test_selection(PerfTaskRunner.tests)

        for widget in self._compose_note_input():
            yield widget

        yield Button("Submit", variant="primary", id="submit-perf-task")

    def on_mount(self) -> None:
        """Called when the form is mounted"""
        self.query_one("#test-list", SelectionList).border_title = "Tests"

    @on(Button.Pressed, "#submit-perf-task")
    def submit_task(self) -> None:
        """Submit the task to the queue"""
        try:
            pipelining: list[int] = self.pipelining.values()
            io_threads: list[int] = self.io_threads.values()
            sizes: list[int] = self.sizes.values()
            warmup: int = self.warmup.value()
            duration: int = self.duration.value()
            tests: list[str] = self.query_one("#test-list", SelectionList).selected
            preload_keys: bool = self.query_one("#preload-keys", Switch).value
            expire_keys: bool = self.query_one("#expire-keys", Switch).value
            profiling: bool = self.query_one("#profiling", Switch).value
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")
            return

        if error := self._validate_tests_selected(tests):
            self.notify(error, severity="error")
            return

        specifiers, note, error = self._validate_and_get_common_inputs()
        if error:
            self.notify(error, severity="error")
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

        tasks: list[BaseTaskData] = []
        for size, pipe, thread, test, specifier in all_tests:
            task = PerfTaskData(
                source=specifier[0],
                specifier=specifier[1],
                replicas=-1,  # TODO configurable replicas
                note=note,
                requirements={},
                val_size=size,
                io_threads=thread,
                pipelining=pipe,
                test=test,
                warmup=warmup,
                duration=duration,
                profiling_sample_rate=profiling_sample_rate,
                has_expire=expire_keys,
                preload_keys=preload_keys,
            )
            tasks.append(task)
        self.queue_tasks(tasks)


class MemTaskForm(BaseTaskForm):
    """Form for creating a performance test task"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sizes = SizesField()

    def compose(self) -> ComposeResult:
        for widget in self._compose_source_specifier_input():
            yield widget

        for widget in self.sizes.widgets():
            yield widget

        yield self._compose_switch_row(
            ("expire-keys", "Expire Keys", False),
        )

        yield self._compose_test_selection(MemTaskRunner.tests)

        for widget in self._compose_note_input():
            yield widget

        yield Button("Submit", variant="primary", id="submit-mem-task")

    def on_mount(self) -> None:
        """Called when the form is mounted"""
        self.query_one("#test-list", SelectionList).border_title = "Tests"

    @on(Button.Pressed, "#submit-mem-task")
    def submit_task(self) -> None:
        """Submit the task to the queue"""
        try:
            sizes: list[int] = self.sizes.values()
            tests: list[str] = self.query_one("#test-list", SelectionList).selected
            expire_keys: bool = self.query_one("#expire-keys", Switch).value
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")
            return

        if error := self._validate_tests_selected(tests):
            self.notify(error, severity="error")
            return

        specifiers, note, error = self._validate_and_get_common_inputs()
        if error:
            self.notify(error, severity="error")
            return

        all_tests = list(
            product(
                tests,
                specifiers,
            )
        )

        tasks: list[BaseTaskData] = []
        for test, specifier in all_tests:
            task = MemTaskData(
                source=specifier[0],
                specifier=specifier[1],
                val_sizes=sizes,
                type=test,
                has_expire=expire_keys,
                replicas=-1,
                note=note,
                requirements={},
            )
            tasks.append(task)
        self.queue_tasks(tasks)


class SyncTaskForm(BaseTaskForm):
    """Form for creating a full sync benchmark task"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.io_threads = IOThreadsField()
        self.sizes = SizesField()
        self.counts = CountsField()

    def compose(self) -> ComposeResult:
        for widget in self._compose_source_specifier_input():
            yield widget

        yield Label("Replicas: 1")  # TODO allow configurable replica count
        yield Label("Test: set")  # TODO allow configurable data type

        for field in (self.io_threads, self.sizes, self.counts):
            for widget in field.widgets():
                yield widget

        yield self._compose_switch_row(
            ("profiling", "Profiling", False),
        )

        for widget in self._compose_note_input():
            yield widget

        yield Button("Submit", variant="primary", id="submit-sync-task")

    @on(Button.Pressed, "#submit-sync-task")
    def submit_task(self) -> None:
        """Submit the task to the queue"""

        replicas = 1
        test = "set"

        try:
            io_threads: list[int] = self.io_threads.values()
            sizes: list[int] = self.sizes.values()
            counts: list[int] = self.counts.values()
            profiling: bool = self.query_one("#profiling", Switch).value
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")
            return

        specifiers, note, error = self._validate_and_get_common_inputs()
        if error:
            self.notify(error, severity="error")
            return

        all_tests = list(
            product(
                sizes,
                counts,
                io_threads,
                specifiers,
            )
        )
        profiling_sample_rate = 3999 if profiling else -1

        tasks: list[BaseTaskData] = []
        for size, count, thread, specifier in all_tests:
            task = SyncTaskData(
                source=specifier[0],
                specifier=specifier[1],
                val_size=size,
                val_count=count,
                io_threads=thread,
                replicas=replicas,
                test=test,
                profiling_sample_rate=profiling_sample_rate,
                note=note,
                requirements={},
            )
            tasks.append(task)
        self.queue_tasks(tasks)


if __name__ == "__main__":
    logging.basicConfig(filename=config.CONDUCTRESS_LOG, encoding="utf-8", level=logging.DEBUG)
    app = BenchmarkApp()
    app.run()
