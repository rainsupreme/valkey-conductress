"""Misc utility functions: Printing, formatting, command execution, etc."""

import os
import signal
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union

from config import SSH_KEYFILE

MILLION = 1_000_000
BILLION = 1_000_000_000

MB = 1024 * 1024
GB = 1024 * MB

MINUTE = 60
HOUR = 60 * MINUTE


def get_console_width(default: int = 80) -> int:
    """Determine the width of the console."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def __fit_number_to_unit(
    number: float, base: Union[int, tuple], units: tuple, decimals: int, threshold: float
) -> str:
    """
    Convert number to a human-readable format with appropriate units.

    Args:
        number (float): number to convert
        base (Union[int, tuple]): base for conversion - int if constant divisor between units (e.g. 1000),
            or tuple of bases for each unit if non-constant (as in time)
        units (tuple): sequence of unit suffixes
        decimals (int): decimals places to show
        threshold (float, optional): Switches to next unit once result in that unit is >= threshold.

    Returns:
        str: formatted string
    """
    number = float(number)
    if isinstance(base, int):
        unit_index = 0
        while number >= base * threshold and unit_index + 1 < len(units):
            unit_index += 1
            number /= base
    else:
        assert len(base) == len(units)
        unit_index = 0
        while unit_index + 1 < len(base) and number >= base[unit_index + 1] * threshold:
            unit_index += 1
            number /= base[unit_index]

    if number.is_integer() or decimals == 0:
        return f"{number:,g}{units[unit_index]}"
    else:
        return f"{number:,.{decimals}f}{units[unit_index]}"


def human(number: float, decimals: int = 1) -> str:
    """Convert a number to a human-readable format."""
    suffixes = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
    return __fit_number_to_unit(number, 1000, suffixes, decimals, threshold=0.5)


def human_byte(number: float, decimals: int = 1) -> str:
    """Convert bytes to human-readable format."""
    suffixes = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    return __fit_number_to_unit(number, 1024, suffixes, decimals, threshold=0.5)


def human_time(number: float) -> str:
    """
    Format a number as time in seconds, minutes, hours, or days

    Args:
        number (float): seconds

    Returns:
        str: formatted string
    """
    divisors = (1, 60, 60, 24)
    units = tuple("smhd")
    return __fit_number_to_unit(number, divisors, units, decimals=0, threshold=1)


def __build_header(left_decor: str, center: str, right_decor: str, fill: str = "─") -> str:
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
    command: Union[str, Sequence[str]],
    remote_ip: Union[str, None] = None,
    remote_pseudo_terminal: bool = True,
    cwd: Union[Path, None] = None,
    check: bool = True,
):
    """Run a console command and return its output."""
    if remote_ip is None:  # local command
        cmd_list = command.split() if isinstance(command, str) else command
        result = subprocess.run(
            cmd_list,
            check=check,
            encoding="utf-8",
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.stderr:
            print(repr(result.stderr))
        return result.stdout, result.stderr

    if isinstance(command, str):
        remote_command = command
    else:
        remote_command = " ".join(command)
    if cwd is not None:
        remote_command = f"cd {str(cwd)}; {remote_command}"

    # remote command
    ssh_command = ["ssh", "-q"]
    if not remote_pseudo_terminal:
        ssh_command += ["-T"]  # disable pseudo-terminal allocation for non-interactive sessions
    ssh_command += ["-i", SSH_KEYFILE, remote_ip, remote_command]

    result = subprocess.run(
        ssh_command, check=check, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout, result.stderr


class RealtimeCommand:
    """Run a local command in real-time and read its output."""

    def __init__(self, command: str, remote: Optional[str] = None):
        self.command = command
        self.remote = remote
        self.p = None

    def __del__(self):
        """Destructor - kill the process if it's still running."""
        self.kill()

    def start(self, ignore_output=False):
        """Start the command in a subprocess."""
        if self.p is not None:
            raise RuntimeError("Command already running")
        command_list = self.command.split()
        if self.remote is not None:
            command_list = ["ssh", "-q", "-i", SSH_KEYFILE, self.remote] + command_list
        output_dest = subprocess.DEVNULL if ignore_output else subprocess.PIPE
        self.p = subprocess.Popen(command_list, stdout=output_dest, stderr=output_dest)
        if not ignore_output:
            assert self.p.stdout is not None
            assert self.p.stderr is not None
            os.set_blocking(self.p.stdout.fileno(), False)
            os.set_blocking(self.p.stderr.fileno(), False)

    def poll_output(self) -> tuple[str, str]:
        """Read output since last poll."""
        if self.p is None:
            raise RuntimeError("Command not started")
        assert self.p.stdout is not None
        assert self.p.stderr is not None
        stdout = self.p.stdout.read()
        stderr = self.p.stderr.read()
        if stdout is not None:
            stdout = stdout.decode("utf-8")
        if stderr is not None:
            stderr = stderr.decode("utf-8")
        return stdout, stderr

    def is_running(self):
        """Check if the process is still running."""
        if self.p is None:
            return False  # process not started
        if self.p.poll() is not None:
            return False  # process finished
        return True

    def interrupt(self):
        """Send interrupt signal to process"""
        if self.p and self.p.poll() is None:
            self.p.send_signal(signal.SIGINT)

    def wait(self):
        """Block until process has ended."""
        if self.p:
            self.p.wait()

    def kill(self):
        """Kill the procress."""
        if self.p:
            self.p.kill()
            self.p.wait()


def hash_file(path: Path):
    """Generate SHA1 hash of a local file."""
    result, _ = run_command(f"sha1sum {str(path)}")
    return result.strip().split()[0]
