"""Misc utility functions: Printing, formatting, command execution, etc."""

import os
import subprocess
from pathlib import Path
from typing import Union

from numerize import numerize

from config import SSH_KEYFILE

MILLION = 1_000_000
MINUTE = 60
HOUR = 60 * MINUTE


def get_console_width(default: int = 80) -> int:
    """Determine the width of the console."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def human(number: float, decimals: int = 2) -> str:
    """Convert a number to a human-readable format."""
    return numerize.numerize(number, decimals=decimals)


def human_byte(number: float) -> str:
    """Convert bytes to human-readable format."""
    number = float(number)
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    unit_index = 0
    while number >= 512 and unit_index + 1 < len(units):
        number /= 1024
        unit_index += 1
    if number.is_integer():
        return f"{number:,g}{units[unit_index]}"
    else:
        return f"{number:,.1f}{units[unit_index]}"


def human_time(number: float) -> str:
    """Convert seconds to human-readable format."""
    number = float(number)
    divisors = [1, 60, 60, 24]
    units = "smhd"
    unit_index = 0
    while unit_index + 1 < len(units) and number >= divisors[unit_index + 1]:
        unit_index += 1
        number /= divisors[unit_index]
    if number.is_integer():
        return f"{number:,g}{units[unit_index]}"
    else:
        return f"{number:,.1f}{units[unit_index]}"


def print_pretty_header(text: str):
    """Print a header with a nice divider."""
    console_width = get_console_width()
    padding = (console_width - len(text)) // 2
    text = " " * padding + text + " " * padding

    endcap = "•"
    center = "•°•♥•°•"
    fill = "─"
    fillsize = (console_width - len(center) - len(endcap) * 2) // 2
    divider = endcap + fill * fillsize + center + fill * fillsize + endcap

    print(divider)
    print(text)


def calc_percentile_averages(data: list, percentages, lowest_vals=False) -> list[float]:
    """Calculate the average of the top or bottom x percents of values in a list."""
    copy = data.copy()
    copy.sort()
    if lowest_vals:
        copy.reverse()

    result = []
    for percent in percentages:
        start_index = len(copy) * (100 - percent) // 100
        section = copy[start_index:]
        section_avg = sum(section) / len(section)
        result.append(section_avg)
    return result


def run_command(
    command: str,
    remote_ip: Union[str, None] = None,
    cwd: Union[Path, None] = None,
    check: bool = True,
):
    """Run a console command and return its output."""
    command_list = command.split()
    if remote_ip is not None:
        command_list = ["ssh", "-i", SSH_KEYFILE, remote_ip] + command_list
        if cwd is not None:
            command_list = ["cd", str(cwd), ";"] + command_list
        result = subprocess.run(command_list, check=check, encoding="utf-8", stdout=subprocess.PIPE)
    else:
        result = subprocess.run(command_list, check=check, encoding="utf-8", cwd=cwd, stdout=subprocess.PIPE)
    return result.stdout


class RealtimeCommand:
    """Run a local command in real-time and read its output."""

    def __init__(self, command: str):
        self.command = command
        self.p = None

    def start(self):
        """Start the command in a subprocess."""
        if self.p is not None:
            raise RuntimeError("Command already running")
        self.p = subprocess.Popen(self.command.split(), stdout=subprocess.PIPE)
        assert self.p.stdout is not None
        os.set_blocking(self.p.stdout.fileno(), False)

    def poll_output(self):
        """Read output since last poll."""
        if self.p is None:
            raise RuntimeError("Command not started")
        assert self.p.stdout is not None
        output = self.p.stdout.read()
        if output is not None:
            output = output.decode("utf-8")
        return output

    def is_running(self):
        """Check if the process is still running."""
        if self.p is None:
            return False  # process not started
        if self.p.poll() is not None:
            return False  # process finished
        return True

    def kill(self):
        """Kill the procress."""
        if self.p:
            self.p.kill()


def hash_file(path: Path):
    """Generate SHA1 hash of a local file."""
    return run_command(f"sha1sum {str(path)}").strip().split()[0]
