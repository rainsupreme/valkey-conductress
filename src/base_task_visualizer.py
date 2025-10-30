"""Base class for task-specific visualizations in the status tab."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Label
from textual_plotext import PlotextPlot


class BaseTaskVisualizer(Container):
    """Base class for task visualizations that can be swapped in the status tab."""

    def __init__(self, task_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id = task_id

    def compose(self) -> ComposeResult:
        """Override to provide task-specific visualization widgets."""
        raise NotImplementedError("Subclasses must implement compose()")

    def refresh_data(self) -> None:
        """Override to update visualization with latest task data."""
        raise NotImplementedError("Subclasses must implement refresh_data()")


class PlaceholderTaskVisualizer(BaseTaskVisualizer):
    """Placeholder visualizer for tasks without a specific implementation."""

    DEFAULT_CSS = """
    PlaceholderTaskVisualizer {
        align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("visualizer not implemented")

    def refresh_data(self) -> None:
        pass


class PlotTaskVisualizer(BaseTaskVisualizer):
    """Base visualizer that renders data into a plot."""

    DEFAULT_CSS = """
    PlotTaskVisualizer {
        height: 1fr;
        width: 1fr;
    }
    #plot-output {
        height: 1fr;
        width: 1fr;
        overflow: hidden;
        padding: 0;
        border: none;
    }
    """

    def format_x_tick(self, value: float) -> str:
        """Override to format x-axis tick labels."""
        return str(value)

    def format_y_tick(self, value: float) -> str:
        """Override to format y-axis tick labels."""
        return str(value)

    def compose(self) -> ComposeResult:
        plot = PlotextPlot()
        plt = plot.plt
        plt.frame(False)
        yield plot
        self._update_plot(plot)

    def _calculate_ticks(
        self, data_min: float, data_max: float, available_space: int, min_spacing: int
    ) -> list[float]:
        """Calculate tick mark positions based on data range and spacing constraints."""
        num_ticks = min(9, available_space // min_spacing) if available_space >= min_spacing else 0
        if num_ticks > 1:
            interval = (data_max - data_min) / (num_ticks - 1)
            return [data_min + i * interval for i in range(num_ticks)]
        return []

    def _update_plot(self, plot: PlotextPlot) -> None:
        """Update plot data and tick marks."""
        data = self.get_plot_data()
        if not data:
            return
        plt = plot.plt
        plt.clear_data()
        plt.scatter(data, marker="braille", color="white")
        xticks = self._calculate_ticks(0, len(data), plot.size.width, 10)
        if xticks:
            plt.xticks(xticks, [self.format_x_tick(x) for x in xticks])
        valid_data = [d for d in data if d is not None]
        if valid_data:
            yticks = self._calculate_ticks(min(valid_data), max(valid_data), plot.size.height, 4)
            if yticks:
                plt.yticks(yticks, [self.format_y_tick(y) for y in yticks])

    def refresh_data(self) -> None:
        plot = self.query_one(PlotextPlot)
        self._update_plot(plot)
        plot.refresh()

    def get_plot_data(self) -> list[float]:
        """Override to provide data points for the scatter plot."""
        raise NotImplementedError("Subclasses must implement get_plot_data()")
