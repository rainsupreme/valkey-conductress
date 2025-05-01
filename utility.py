"""Misc utility functions: Printing, formatting, command execution, etc."""

import os
import subprocess
from pathlib import Path
from typing import Union

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


def __fit_number_to_unit(number: float, base: Union[int, tuple], units: tuple, decimals: int) -> str:
    """Fit a number to a unit."""
    number = float(number)
    if isinstance(base, int):
        unit_index = 0
        while number >= base / 2 and unit_index + 1 < len(units):
            unit_index += 1
            number /= base
    else:
        assert len(base) == len(units)
        unit_index = 0
        while unit_index + 1 < len(base) and number >= base[unit_index + 1] / 2:
            unit_index += 1
            number /= base[unit_index]

    if number.is_integer():
        return f"{number:,g}{units[unit_index]}"
    else:
        return f"{number:,.{decimals}f}{units[unit_index]}"


def human(number: float, decimals: int = 1) -> str:
    """Convert a number to a human-readable format."""
    suffixes = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
    return __fit_number_to_unit(number, 1000, suffixes, decimals)


def human_byte(number: float, decimals: int = 1) -> str:
    """Convert bytes to human-readable format."""
    suffixes = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    return __fit_number_to_unit(number, 1024, suffixes, decimals)


def human_time(number: float, decimals: int = 1) -> str:
    """Convert seconds to human-readable format."""
    number = float(number)
    divisors = (1, 60, 60, 24)
    units = tuple("smhd")
    return __fit_number_to_unit(number, divisors, units, decimals)


def __build_header(left_decor: str, center: str, right_decor: str, fill: str = "─"):
    console_width = get_console_width()

    # Decorative elements
    left_decor = "⊹˚₊‧"
    right_decor = "‧₊˚⊹"
    fill = "─"

    # Calculate padding needed
    content_len = len(center) + 2  # +2 for spaces around text
    decor_len = len(left_decor) + len(right_decor)
    fill_len = console_width - content_len - decor_len

    # Split fill chars evenly on each side
    fill_str = fill * (fill_len // 2)

    # Construct and print the header
    header = f"{left_decor}{fill_str} {center} {fill_str}{right_decor}"
    return header


def print_inline_header(text: str):
    """Print decorative header with inline centered text"""
    print(__build_header("⊹˚₊‧", text, "‧₊˚⊹", "─"))


def print_pretty_divider():
    """Print a decorative header (no text)"""
    print(__build_header("⊹˚₊‧", "•°•♥•°•", "‧₊˚⊹", "─"))


def print_centered_text(text: str):
    """Print text centered in the console."""
    console_width = get_console_width()
    padding = (console_width - len(text)) // 2
    text = " " * padding + text + " " * padding
    print(text)


def print_pretty_header(text: str):
    """Print a header with a nice divider."""
    print_pretty_divider()
    print_centered_text(text)


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
    command: Union[str, list[str]],
    remote_ip: Union[str, None] = None,
    remote_pseudo_terminal: bool = True,
    cwd: Union[Path, None] = None,
    check: bool = True,
):
    """Run a console command and return its output."""
    if isinstance(command, str):
        command_list = command.split()
    else:
        command_list = command

    if remote_ip is None:  # local command
        result = subprocess.run(command_list, check=check, encoding="utf-8", cwd=cwd, stdout=subprocess.PIPE)
        return result.stdout

    # remote command
    ssh_command = ["ssh", "-q"]
    if not remote_pseudo_terminal:
        ssh_command += ["-T"]  # disable pseudo-terminal allocation for non-interactive sessions
    ssh_command += ["-i", SSH_KEYFILE, remote_ip]

    if cwd is not None:
        command_list = ["cd", str(cwd), ";"] + command_list

    result = subprocess.run(ssh_command + command_list, check=check, encoding="utf-8", stdout=subprocess.PIPE)
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
