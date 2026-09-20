<picture>
  <source media="(prefers-color-scheme: dark)" srcset="brand/mono/conductress-lockup-mono-light-horizontal.svg">
  <img src="brand/mono/conductress-lockup-mono-dark-horizontal.svg" width="400" alt="Conductress">
</picture>

# Contributing to Conductress

Thanks for contributing. This guide covers the development setup, the checks CI runs, and the conventions this project follows.

## Development setup

Conductress is a Python package (requires Python 3.9 or newer). Work in a virtualenv with an editable install:

```bash
git clone https://github.com/rainsupreme/valkey-conductress.git
cd valkey-conductress
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,control,plots]'
```

The `dev` extra provides the test and lint tooling (pytest, black, isort, pylint, mypy, flake8, hypothesis). `control` and `plots` pull in the control-service and plotting dependencies. Installing all three matches what CI installs for the test job.

### Developer checkout vs. runner provisioning

Two commands look alike and do different things:

- **`make install`** (or the `pip install -e` above) sets up a checkout for working on the code. It touches only the active virtualenv. This is all you need to run the tests and the linters.
- **`conductress setup`** turns a machine into a benchmark runner. For this host and every host in `servers.json` it runs `sudo` to upgrade and install system packages, raises file-descriptor limits in `/etc/security/limits.conf`, enables io_uring via sysctl, clones and builds the load generators, and installs a systemd service. It expects the checkout at `~/conductress`. Do not run it on a development machine unless you want that machine to become a runner.

The Makefile never calls `conductress setup`, and `setup` never calls `make`; the one thing both do is the editable install.

## Running the checks

CI (`.github/workflows/tests.yml`) runs the checks below. Run them locally before pushing — the fastest way is `make ci` (see [Makefile targets](#makefile-targets)), which mirrors these exactly.

The lint job pins `black==25.11.0` and `isort==5.13.2`. If you installed the `dev` extra (which allows older black/isort), overlay the pinned versions so your local run matches CI:

```bash
pip install 'black==25.11.0' 'isort==5.13.2'
```

Then:

```bash
# Lint
black --check --line-length 120 src/ tests/
isort --check-only --profile black --line-length 120 src/ tests/
pylint --errors-only --disable=import-error src/

# Tests (unit + control service) with the coverage gate
PYTHONPATH=src pytest tests/unit tests/control --cov=conductress --cov-fail-under=72

# Integration tests that do not need a running server
PYTHONPATH=src pytest tests/integration -m "not requires_server"

# Type checking
mypy src/ --ignore-missing-imports
```

To auto-format before committing:

```bash
black --line-length 120 src/ tests/
isort --profile black --line-length 120 src/ tests/
```

### Makefile targets

The Makefile wraps these checks and acts on the active virtualenv only. Each target runs against `src/` and `tests/`:

- `make install` — editable install with the `dev`, `control`, and `plots` extras.
- `make lint` — `black --check`, `isort --check-only`, and `pylint --errors-only`.
- `make format` — apply `black` and `isort` in place.
- `make test` — unit and control-service tests with the coverage gate.
- `make integration` — integration tests that do not need a running server.
- `make typecheck` — `mypy` over `src/`.
- `make ci` — `lint`, `test`, `integration`, and `typecheck` in sequence, mirroring what CI runs.

### Always set `PYTHONPATH=src`

Run pytest and mypy with `PYTHONPATH=src`. Other editable installs on the same host can otherwise shadow this tree, so a check may run against the wrong code. The Makefile sets this for you.

## Sign off your commits (DCO)

Every commit must carry a `Signed-off-by` trailer. The `dco` job in CI rejects any commit in a pull request that lacks one. Add it with the `-s` flag:

```bash
git commit -s -m "docs: fix the quick-start install command"
```

The trailer certifies you have the right to submit the contribution under the project's license (the [Developer Certificate of Origin](https://developercertificate.org/)). Configure your Git `user.name` and `user.email` so the trailer identifies you.

## Branch and pull request conventions

Recent history (`git log --oneline`) shows the conventions this project follows:

- **Conventional-commit subjects.** Prefix the subject with the change type: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `ci:`, `perf:`, `test:`, or `build:`. Add a scope in parentheses when it helps, e.g. `fix(replica-read): ...`.
- **One logical change per branch.** Branch names use the same prefixes, e.g. `docs/contributing-and-makefile`, `feature/replica-read-task`, `fix/storm-plot-columns`.
- **Squash merge.** Pull requests are squash-merged, so the merge commit's subject carries the PR number, e.g. `fix: skip hit-rate prefill guard for workloads with no GETs (#184)`. Keep the PR title in conventional-commit form since it becomes that subject.
- Open pull requests against the `main` branch.

## Write durable language

Anything committed to the repository should still read correctly a year from now, to someone who was not part of the conversation that produced it. Code, comments, docstrings, help text, test names, and commit messages describe what the code does and why, not the circumstances under which it was written.

Avoid, in code and comments:

- **Time anchors:** "currently", "today", "for now", "recently", "the new ...", "legacy" (name the thing instead), dates, "as of" qualifiers.
- **Process references:** pull request or issue numbers, review threads, audits, phases or steps of a plan, the name of the branch or worker that produced the change.
- **Specific deployments:** hostnames, fleet names, a particular runner's configuration, "the live fleet", "what production uses". Defaults are described as defaults, not as a snapshot of some installation.
- **Specific measurements or experiments:** a particular benchmark run, workload, test campaign, or result. If a number justifies a design choice, state the mechanism and the order of magnitude, not the run.
- **People:** who asked for or wrote something. Git history carries authorship.

Prefer the form that stays true: "`--source` filters both sides" rather than "`--source` currently filters both sides"; "guards against an accidental trim of the built-in repository list" rather than "the fleet queues with `--source valkey-rainfall`, so keep this entry".

Documents under `docs/` get a little more leeway, since a design note or plan is by nature a point-in-time record. Even there, lean durable: describe the system and the reasoning, and keep references to a specific discussion, workload, test run, or deployment out of the text unless the document is explicitly a record of that event (say so in its first paragraph if it is).

Commit subjects and bodies describe the change and its motivation in the same terms. The squash-merge subject carries the PR number automatically; the body does not need to repeat it.

## Runtime state is not source

When a runner executes tasks it writes per-host runtime state into the project root: `sweep_data/`, `benchmark_queue/`, `results/` (including `results/output.jsonl`), `tmp/`, `log.txt`, `failed/`, `failed_tasks.jsonl`, and similar. Local configuration (`runner.json`, `servers.json`, `server-keyfile.pem`) also lives there. All of it is git-ignored and must never be committed. If `git status` shows any of these, do not add them — check `.gitignore` before staging.

## Questions

Open an issue or start a discussion on the repository if something here is unclear or out of date.
