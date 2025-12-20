"""Conductress TUI application for Valkey benchmarking tasks."""

import logging
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable, Iterator, Optional, TypeVar, Union

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.theme import Theme
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
from src.tasks.task_perf_benchmark import (
    PerfTaskData,
    PerfTaskRunner,
    PerfTaskVisualizer,
)
from src.tui_data_service import TUIDataService

from . import config
from .base_task_visualizer import BaseTaskVisualizer, PlaceholderTaskVisualizer
from .task_queue import BaseTaskData, TaskQueue
from .utility import HumanByte, HumanNumber, HumanTime

logger = logging.getLogger(__name__)


class ConfirmCancelScreen(ModalScreen[tuple[bool, str]]):
    """Screen with a dialog to confirm task cancellation."""

    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f"Cancel task {self.task_id}?", id="question")
            with Horizontal():
                yield Button("Yes", variant="warning", id="yes")
                yield Button("No", variant="primary", id="no")

    @on(Button.Pressed, "#yes")
    def confirm(self) -> None:
        self.dismiss((True, self.task_id))

    @on(Button.Pressed, "#no")
    def cancel(self) -> None:
        self.dismiss((False, self.task_id))


class BenchmarkApp(App):
    """Main application class for the benchmark app."""

    TITLE = "Valkey Conductress"
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
    #status-table {
        height: auto;
        max-height: 7;
    }
    #task-visualizer-container {
        height: 1fr;
    }
    ConfirmCancelScreen {
        align: center middle;
    }
    ConfirmCancelScreen > Vertical {
        width: 50;
        height: auto;
        background: $panel;
        border: thick $primary;
        padding: 1 2;
    }
    ConfirmCancelScreen #question {
        width: 100%;
        content-align: center middle;
        padding: 1;
    }
    ConfirmCancelScreen Horizontal {
        width: 100%;
        height: auto;
        align: center middle;
    }
    ConfirmCancelScreen Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(name="Valkey Conductress")
        with TabbedContent(initial="tab-status", id="root-tabs"):
            with TabPane("Status", id="tab-status"):
                yield Static("Last update: Never", id="status-table-status")
                yield DataTable(id="status-table", cursor_type="row")
                yield ScrollableContainer(id="task-visualizer-container")
            with TabPane("Queue", id="tab-queue"):
                yield Static("Last update: Never", id="queue-table-status")
                yield DataTable(id="queue-table", cursor_type="row")
                yield Button("Remove Selected Task", variant="warning", id="remove-queue-task")
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
        custom_theme = Theme(
            name="Conductress",
            primary="#ccccff",
            secondary="#ffcc99",
            accent="#ff6b9d",
            warning="#ffaa00",
            error="#ff4444",
            success="#44ff88",
            background="#111111",
            surface="#000000",
            panel="#1a1a1a",
            foreground="#ccccff",
        )
        self.register_theme(custom_theme)
        self.theme = "Conductress"
        self.data_service = TUIDataService()
        self.current_visualizer: Optional[BaseTaskVisualizer] = None
        self.previous_status_count = 0
        self.current_task_id: Optional[str] = None
        self.queue_tasks: list[BaseTaskData] = []
        self.set_interval(config.TUI_REFRESH_INTERVAL, self.refresh_data)
        self.refresh_data()

    @on(TabbedContent.TabActivated, "#root-tabs")
    def on_tab_changed(self, event: TabbedContent.TabActivated) -> None:
        """Handle tab change to update content."""
        self.refresh_data()

    @on(DataTable.RowSelected, "#status-table")
    def on_status_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle status table row selection."""
        table = event.data_table
        row_key = event.row_key
        task_id = str(table.get_row(row_key)[0])
        self._swap_visualizer(task_id)

    @on(Button.Pressed, "#remove-queue-task")
    def on_remove_queue_task(self) -> None:
        """Handle remove queue task button press."""
        table = self.query_one("#queue-table", DataTable)
        if table.cursor_row is not None:
            row = table.get_row_at(table.cursor_row)
            task_id = str(row[0])
            self.push_screen(ConfirmCancelScreen(task_id), self._remove_queue_task)

    def _remove_queue_task(self, result: tuple[bool, str]) -> None:
        """Remove a task from the queue."""
        confirmed, task_id = result
        if not confirmed:
            return
        self.run_worker(lambda: self._remove_task_worker(task_id), name="_remove_task_worker", thread=True)

    def _remove_task_worker(self, task_id: str) -> bool:
        """Worker to remove task from queue."""
        return self.data_service.remove_task(task_id)

    def _swap_visualizer(self, task_id: str) -> None:
        """Swap the task visualizer for the selected task."""
        self.run_worker(
            lambda: self._swap_visualizer_worker(task_id),
            name="_swap_visualizer_worker",
            exclusive=True,
            thread=True,
        )

    def _swap_visualizer_worker(self, task_id: str):
        """Worker to fetch task status for visualizer."""
        return self.data_service.get_task_status(task_id), task_id

    def _mount_visualizer(self, task_id: str, status) -> None:
        """Mount the appropriate visualizer based on task status."""
        container = self.query_one("#task-visualizer-container", ScrollableContainer)
        container.remove_children()

        if status and status.task_type:
            task_category = status.task_type.split("-")[0]
            if task_category == "perf":
                protocol = FileProtocol(task_id, "client")
                self.current_visualizer = PerfTaskVisualizer(task_id, protocol)
            else:
                self.current_visualizer = PlaceholderTaskVisualizer(task_id)
        else:
            self.current_visualizer = PlaceholderTaskVisualizer(task_id)

        container.mount(self.current_visualizer)

    def refresh_data(self) -> None:
        """Refresh the table data."""
        self.run_worker(self._refresh_worker, exclusive=True, thread=True)

    def _refresh_worker(self) -> tuple:
        """Worker to fetch all data in background."""
        tasks, active_tasks = self.data_service.refresh_all()
        return tasks, active_tasks

    def on_worker_state_changed(self, event) -> None:
        """Handle worker completion."""
        if not event.worker.is_finished or event.worker.result is None:
            return

        if event.worker.name == "_refresh_worker":
            tasks, active_tasks = event.worker.result
            self._update_ui(tasks, active_tasks)
        elif event.worker.name == "_remove_task_worker":
            if event.worker.result:
                logger.info("Task removed successfully")
                self.refresh_data()
            else:
                logger.warning("Failed to remove task")
        elif event.worker.name == "_swap_visualizer_worker":
            status, task_id = event.worker.result
            self._mount_visualizer(task_id, status)

    def _update_ui(self, tasks: list[BaseTaskData], active_tasks: dict) -> None:
        """Update all UI elements with refreshed data."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        tabs = self.query_one("#root-tabs", TabbedContent)
        active_tab = tabs.active

        tabs.get_tab("tab-queue").label = f"Queue ({len(tasks)})"
        tabs.get_tab("tab-status").label = f"Status ({len(active_tasks)})"
        self.queue_tasks = tasks

        if active_tab == "tab-queue":
            self._populate_table(
                "#queue-table",
                ["Task ID", "Status", "Type", "Source:Specifier", "Description", "Note"],
                self._queue_rows(tasks, active_tasks),
            )
            self.query_one("#queue-table-status", Static).update(f"Last polled: {timestamp}")
        elif active_tab == "tab-status":
            self._populate_table(
                "#status-table", ["Task ID", "State", "PID", "Progress"], self._status_rows(active_tasks)
            )
            self.query_one("#status-table-status", Static).update(f"Last update: {timestamp}")
            self._update_visualizer(active_tasks)
            if self.current_visualizer:
                self.current_visualizer.refresh_data()

    def _populate_table(self, table_id: str, columns: list[str], rows: list[tuple]) -> None:
        """Populate a table with data, preserving cursor position."""
        table = self.query_one(table_id, DataTable)
        cursor_row = table.cursor_row
        table.clear()

        if not table.columns:
            table.add_columns(*columns)

        for row in rows:
            table.add_row(*row)

        if cursor_row is not None and cursor_row < len(rows):
            table.move_cursor(row=cursor_row)

    def _queue_rows(self, tasks: list[BaseTaskData], active_tasks: dict) -> list[tuple]:
        """Generate queue table rows."""
        rows = []
        for task in tasks:
            task_type = task.task_type.removesuffix("TaskData")
            task_status = ""
            if task.task_id in active_tasks:
                status = active_tasks[task.task_id]
                progress = status.steps_completed / status.steps_total if status.steps_total else 0
                task_status = f"{status.state} {progress*100:.0f}%"
            rows.append(
                (
                    task.task_id,
                    task_status,
                    task_type,
                    f"{task.source}:{task.specifier}",
                    task.short_description(),
                    task.note,
                )
            )
        return rows

    def _status_rows(self, active_tasks: dict) -> list[tuple]:
        """Generate status table rows."""
        rows = []
        for task_id, status in active_tasks.items():
            progress = "N/A"
            if status.steps_total and status.steps_completed is not None:
                pct = (status.steps_completed / status.steps_total) * 100
                progress = f"{pct:.0f}% ({status.steps_completed}/{status.steps_total})"
            rows.append((task_id, status.state, str(status.pid or "N/A"), progress))
        return rows

    def _update_visualizer(self, active_tasks: dict) -> None:
        """Update visualizer based on active tasks."""
        running_task_ids = list(active_tasks.keys())
        if self.current_task_id and self.current_task_id not in running_task_ids:
            self.current_task_id = running_task_ids[0] if running_task_ids else None
            if self.current_task_id:
                self.call_after_refresh(self._swap_visualizer, self.current_task_id)
        elif not self.current_task_id and running_task_ids:
            self.current_task_id = running_task_ids[0]
            self.call_after_refresh(self._swap_visualizer, self.current_task_id)
        self.previous_status_count = len(active_tasks)


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

    def _compose_make_args_input(self) -> Iterator[Union[Label, Input]]:
        """Yield make args input widgets"""
        yield Label("Make Args (optional)", classes="form-label")
        yield Input(
            value=config.DEFAULT_MAKE_ARGS,
            placeholder="OPTIMIZATION=-O2",
            id="make-args",
            classes="form-input",
        )

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

    def _validate_and_get_common_inputs(self) -> tuple[list[tuple[str, str]], str, str, Optional[str]]:
        """Validate and return (specifiers, note, make_args, error_message)"""
        source_specifier_list = self.query_one("#specifiers", Input).value
        note = self.query_one("#note", Input).value.strip()
        make_args = self.query_one("#make-args", Input).value.strip()

        specifiers, error = SourceSpeciferValidator.parse_source_specifier_list(source_specifier_list)
        if error:
            return [], "", "", f"source:specifier list: {error}"

        return specifiers, note, make_args, None

    def _validate_tests_selected(self, tests: list[str]) -> Optional[str]:
        """Return error message if no tests selected, None otherwise"""
        if not tests:
            return "No tests selected"
        return None

    def queue_tasks(self, tasks: list[BaseTaskData]) -> None:
        """Queue the tasks and update the view"""
        for task in tasks:
            self.queue.submit_task(task)
        self.app.call_later(self.update_queue_view)
        self.notify(f"{len(tasks)} tasks queued 🌧 ♥")


class PerfTaskForm(BaseTaskForm):
    """Form for creating a performance test task"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipelining = PipeliningField()
        self.io_threads = IOThreadsField()
        self.sizes = SizesField()
        self.warmup = NumberField("Warmup (seconds)", "warmup", "1m", "1m", HumanTime)
        self.duration = NumberField("Duration (seconds)", "duration", "15m", "15m", HumanTime)

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
            ("perf-stat", "Perf Stat", False),
        )

        yield self._compose_test_selection(PerfTaskRunner.tests)

        for widget in self._compose_note_input():
            yield widget

        for widget in self._compose_make_args_input():
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
            perf_stat: bool = self.query_one("#perf-stat", Switch).value
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")
            return

        if error := self._validate_tests_selected(tests):
            self.notify(error, severity="error")
            return

        specifiers, note, make_args, error = self._validate_and_get_common_inputs()
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
                make_args=make_args,
                val_size=size,
                io_threads=thread,
                pipelining=pipe,
                test=test,
                warmup=warmup,
                duration=duration,
                profiling_sample_rate=profiling_sample_rate,
                perf_stat_enabled=perf_stat,
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

        for widget in self._compose_make_args_input():
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

        specifiers, note, make_args, error = self._validate_and_get_common_inputs()
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
                make_args=make_args,
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

        for widget in self._compose_make_args_input():
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

        specifiers, note, make_args, error = self._validate_and_get_common_inputs()
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
                make_args=make_args,
            )
            tasks.append(task)
        self.queue_tasks(tasks)


if __name__ == "__main__":
    logging.basicConfig(filename=config.CONDUCTRESS_LOG, encoding="utf-8", level=logging.DEBUG)
    app = BenchmarkApp()
    app.run()
