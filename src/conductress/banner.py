"""The `conductress --version` banner: the brand lockup on a terminal, a plain version line anywhere else.

The lockup is pre-rendered ANSI text shipped as package data. It is generated from the
brand kit, not edited by hand::

    python3 brand/terminal/conductress-logo-ascii.py --palette sunset --size 24 --no-labels \\
        > src/conductress/assets/lockup-24.ansi

Regenerate it whenever the mark changes so the banner and ``brand/`` agree.
"""

import argparse
import os
import sys
from importlib import metadata, resources
from typing import IO, Any, Mapping, Optional, Sequence, Union

PACKAGE = "conductress"


def package_version() -> str:
    """The installed package version, or ``unknown`` when running from a source tree that is not installed."""
    try:
        return metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:
        return "unknown"


def version_line() -> str:
    return f"{PACKAGE} {package_version()}"


def load_lockup() -> str:
    """The 24-bit ANSI lockup, without trailing blank lines."""
    path = resources.files(PACKAGE).joinpath("assets").joinpath("lockup-24.ansi")
    return path.read_text(encoding="utf-8").rstrip("\n")


def wants_lockup(stream: IO[str], env: Mapping[str, str]) -> bool:
    """Show the lockup only where it renders: an interactive terminal that has not asked for plain output.

    ``NO_COLOR`` (any value, per https://no-color.org) and ``TERM=dumb`` both select the plain line, and so
    does a pipe or file, so ``conductress --version | cut -d' ' -f2`` keeps working.
    """
    if "NO_COLOR" in env or env.get("TERM", "") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def render_version(stream: IO[str], env: Mapping[str, str]) -> str:
    """The text ``--version`` writes to ``stream``: lockup plus version line, or the version line alone."""
    if wants_lockup(stream, env):
        return f"{load_lockup()}\n\n{version_line()}\n"
    return f"{version_line()}\n"


def print_version(stream: Optional[IO[str]] = None, env: Optional[Mapping[str, str]] = None) -> None:
    out = sys.stdout if stream is None else stream
    out.write(render_version(out, os.environ if env is None else env))
    out.flush()


class VersionAction(argparse.Action):
    """``--version`` flag that prints the banner and exits, like argparse's built-in version action."""

    def __init__(self, option_strings: Sequence[str], dest: str, **kwargs: Any) -> None:
        kwargs.setdefault("nargs", 0)
        kwargs.setdefault("help", "print the version and exit")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Union[str, Sequence[Any], None],
        option_string: Optional[str] = None,
    ) -> None:
        print_version()
        parser.exit(0)
