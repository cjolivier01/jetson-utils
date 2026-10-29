# Repository Guidelines

## Worktrees
- Start new work in a Git worktree that is a sibling of this checkout, never inside it: `git worktree add -b <task> ../jetson-utils-<task>`.
- If this checkout is already a worktree, keep the work here rather than nesting another one. Check with `[ "$(git rev-parse --git-dir)" != "$(git rev-parse --git-common-dir)" ]`.
- Leave the primary checkout (usually on `master`) unmodified; use it to fetch, compare revisions and review. An explicit instruction to work elsewhere wins.
- There is no `.bazelignore` here, so a worktree anywhere under the repository root adds a second set of `BUILD.bazel` files to `//...` and breaks `bazel build //...`.
- Bazel keys its output base on the workspace path, so each worktree builds cold and keeps its own `bazel-*` symlinks.
- Remove the worktree once its branch merges (`git worktree remove ../jetson-utils-<task>`), and run `git worktree prune` if the directory was deleted by hand, otherwise the stale entry keeps its branch locked.

## Project Structure & Module Organization
- C++/CUDA core lives under `base/`, `camera/`, `codec/`, `cuda/`, `display/`, `image/`, `input/`, `network/`, `threads/`, `video/`.
- Bazel build files are per-module (`BUILD.bazel`) with a root `WORKSPACE` and helper rules in `bazel/` and `buildfiles/third_party/`.
- Python bindings and examples are under `python/` (`python/bindings/` for pybind sources; `python/examples/`).
- CMake files exist for legacy/alternative builds but Bazel is the primary flow here.

## Build, Test, and Development Commands
- Build all targets (debug): `./bld` (wraps `bazelisk build --config=debug //...`).
- Build with Bazel directly: `bazel build //...` or narrow like `bazel build //python/bindings:all`.
- Run tests (if present): `bazel test //...`.
- Example run (after build): execute binaries from Bazel runfiles or `bazel run //video:video_output_demo` (adjust to target).
- CMake alternative: `mkdir build && cd build && cmake .. && make -j$(nproc)`.
- Build Python wheel: `bazel run //python:bdist_wheel` (emits a `.whl` into top-level `dist/`).

## Coding Style & Naming Conventions
- C/C++/CUDA: C++17, `-fPIC`. Indentation uses tabs (see `.editorconfig`: `indent_style = tab`, `indent_size = 5`).
- Headers: `.h`/`.hpp`; inline helpers may use `.inl`.
- Targets: prefer descriptive Bazel target names (e.g., `:video_output`, `:image_io`).
- Python examples follow standard PEP 8 where practical; no enforced linter yet.

## Testing Guidelines
- Use Bazel tests: `cc_test` for C++ and `py_test` for Python.
- Place tests alongside modules or under a `tests/` package with clear target names like `:logging_test`, `:image_io_test`.
- Aim for fast, hermetic tests; gate hardware/GPU integration behind flags.
- Example: run `bazel test //base:logging_test`.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood, concise summary (≤72 chars), followed by detail when needed.
- Reference issues (`Fixes #123`) and note platforms affected (Jetson model, x86_64).
- PRs: include description, build/test steps, and screenshots/logs when relevant. Call out CUDA/GStreamer or ABI changes.

## Environment & Configuration
- CUDA toolchains auto-detected via `rules_cuda`. System deps (GStreamer, GLib, OpenCV) are wired via `buildfiles/third_party/`.
- Python linking uses a `conda_repository` rule for `libpython` (respects `CONDA_PREFIX`). Ensure your env is active when building.
