load("@bazel_tools//tools/build_defs/repo:utils.bzl", "workspace_and_buildfile")

# Returns True if the given string ends with a slash.
def _ends_with_slash(s):
    if s == "":
        return False
    return s[len(s) - 1:] == "/"

def _basename(path):
    parts = path.rsplit("/", 1)
    if len(parts) == 1:
        return parts[0]
    return parts[1]

def _path_exists(ctx, path):
    return ctx.path(path).exists

def _first_existing(ctx, paths):
    for path in paths:
        if _path_exists(ctx, path):
            return path
    return ""

def _find_versioned_file(ctx, directory, prefix):
    dir_path = ctx.path(directory)

    if not dir_path.exists:
        return ""

    for entry in dir_path.readdir():
        entry_str = str(entry)
        name = _basename(entry_str)

        if name.startswith(prefix + "."):
            return entry_str

    return ""

# Implementation of the conda repository rule.
def conda_repo_setup(ctx):
    # Get the conda installation root.
    conda_root = ctx.os.environ.get("CONDA_PREFIX")
    if not conda_root:
        fail("Environment variable CONDA_PREFIX is not set.")

    # Retrieve rule attributes.
    pkg_dir = ctx.attr.package_dir
    file_list = ctx.attr.files
    prefix_to_strip = ctx.attr.strip_prefix

    # If a package directory is provided, symlink its content recursively.
    if pkg_dir:
        if _ends_with_slash(pkg_dir):
            fail("Please remove the trailing slash from 'package_dir': " + pkg_dir)

        # Build the absolute path to the package.
        pkg_absolute = conda_root + "/" + pkg_dir
        # Default: strip off the conda root and the following slash.
        num_chars_to_strip = len(conda_root + "/")

        if prefix_to_strip:
            if _ends_with_slash(prefix_to_strip):
                fail("Please remove the trailing slash from 'strip_prefix': " + prefix_to_strip)
            alt_pkg_path = conda_root + "/" + prefix_to_strip
            if not ctx.path(alt_pkg_path).exists:
                fail("Cannot locate the path derived from 'strip_prefix': " + alt_pkg_path)
            if alt_pkg_path.find(pkg_absolute) != 0:
                fail("Expected 'strip_prefix' (" + alt_pkg_path +
                     ") to be under the package directory (" + pkg_absolute + ")")
            pkg_absolute = alt_pkg_path
            num_chars_to_strip = len(alt_pkg_path + "/")

        # Iterate over all entries in the package directory and symlink them.
        for entry in ctx.path(pkg_absolute).readdir():
            # Compute the destination path by removing the leading portion.
            dest = str(entry)[num_chars_to_strip:]
            ctx.symlink(entry, dest)

    # Symlink any individual files specified.
    for f in file_list:
        abs_file = conda_root + "/" + f
        if not ctx.path(abs_file).exists:
            fail("File not found: " + abs_file)
        ctx.symlink(abs_file, f)

    # Finally, generate the BUILD and WORKSPACE files in this repository.
    workspace_and_buildfile(ctx)

# The repository rule "conda_repository" creates a repository by symlinking
# content from your conda environment (under CONDA_PREFIX). It can either symlink
# an entire directory (via "package_dir") with an optional "strip_prefix" or
# individual files (via "files").
conda_repository = repository_rule(
    doc = """
        Creates a repository from a Conda environment by symlinking files and/or
        a directory located relative to CONDA_PREFIX.

        The 'package_dir' attribute (if provided) causes the entire directory's
        contents to be symlinked into the repository. An optional 'strip_prefix'
        can adjust the subdirectory layout.

        Additionally, individual files listed in 'files' are symlinked directly.

        Example usage in WORKSPACE:

        load("//:conda_repo.bzl", "conda_repository")
        conda_repository(
            name = "conda_lzma",
            package_dir = "include/lzma",
            files = [
                "include/lzma.h",
                "lib/liblzma.so"
            ],
            build_file = "@//third_party/bazel-builds:conda_lzma.BUILD",
        )
    """,
    implementation = conda_repo_setup,
    attrs = {
        "package_dir": attr.string(),
        "files": attr.string_list(),
        "build_file": attr.label(allow_single_file = True),
        "build_file_content": attr.string(),
        "workspace_file": attr.label(allow_single_file = True),
        "workspace_file_content": attr.string(),
        "strip_prefix": attr.string(),
    },
    environ = ["CONDA_PREFIX"],
)

def _discover_root(ctx, env_vars, fixed_candidates, scan_candidates, validator):
    for env_var in env_vars:
        value = ctx.os.environ.get(env_var)
        if value and validator(ctx, value):
            return value

    for candidate in fixed_candidates:
        if validator(ctx, candidate):
            return candidate

    for scan_dir, prefix in scan_candidates:
        scan_path = ctx.path(scan_dir)

        if not scan_path.exists:
            continue

        for entry in scan_path.readdir():
            entry_str = str(entry)

            if not _basename(entry_str).startswith(prefix):
                continue

            if validator(ctx, entry_str):
                return entry_str

    return ""

def _symlink_entries(ctx, root, entries):
    for entry in entries:
        path = root + "/" + entry
        if ctx.path(path).exists:
            ctx.symlink(ctx.path(path), entry)

def _valid_cuda_root(ctx, root):
    if not root:
        return False

    if not _path_exists(ctx, root + "/bin/nvcc"):
        return False

    for rel in [
        "targets/x86_64-linux/lib/libculibos.a",
        "targets/aarch64-linux/lib/libculibos.a",
        "targets/sbsa-linux/lib/libculibos.a",
    ]:
        if _path_exists(ctx, root + "/" + rel):
            return True

    return False

def _valid_rocm_root(ctx, root):
    if not root:
        return False

    for rel in [
        "bin/hipcc",
        "include/hip/hip_runtime.h",
    ]:
        if not _path_exists(ctx, root + "/" + rel):
            return False

    return bool(_find_rocm_runtime_library(ctx, root))

def _find_rocm_runtime_library(ctx, root):
    runtime = _first_existing(ctx, [
        root + "/lib/libamdhip64.so",
        root + "/lib64/libamdhip64.so",
    ])

    if runtime:
        return runtime

    for directory in [root + "/lib", root + "/lib64"]:
        runtime = _find_versioned_file(ctx, directory, "libamdhip64.so")
        if runtime:
            return runtime

    return ""

def _local_cuda_sdk_repo_impl(ctx):
    root = _discover_root(
        ctx,
        ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"],
        ["/usr/local/cuda"],
        [("/usr/local", "cuda-")],
        _valid_cuda_root,
    )

    build = [
        'load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")',
        'package(default_visibility = ["//visibility:public"])',
    ]

    if root:
        # Cross toolkits can contain both host and target libraries. Select the
        # archive using Bazel's target CPU instead of the first existing path.
        culibos = {}
        for cpu, triples in [
            ("x86_64", ["x86_64-linux"]),
            ("aarch64", ["aarch64-linux", "sbsa-linux"]),
        ]:
            for triple in triples:
                rel = "targets/%s/lib/libculibos.a" % triple
                if ctx.path(root + "/" + rel).exists:
                    build.append(
                        'config_setting(name = "%s", constraint_values = ["@platforms//cpu:%s"])' % (cpu, cpu),
                    )
                    culibos[":" + cpu] = rel
                    break

        if not culibos:
            fail("Found CUDA toolkit at %s but could not locate libculibos.a" % root)

        _symlink_entries(ctx, root, ["bin", "extras", "lib64", "nvvm", "targets"])

        build.extend([
            'filegroup(name = "nvcc", srcs = ["bin/nvcc"])',
            'cc_import(name = "culibos", static_library = select(%s, no_match_error = "CUDA toolkit has no libculibos.a for the selected target CPU"))' % repr(culibos),
        ])
    else:
        build.extend([
            'filegroup(name = "nvcc", srcs = [])',
            'cc_library(name = "culibos", srcs = [], hdrs = [])',
        ])

    ctx.file("WORKSPACE", 'workspace(name = "%s")\n' % ctx.name)
    ctx.file("BUILD.bazel", "\n".join(build) + "\n")

def _local_rocm_sdk_repo_impl(ctx):
    root = _discover_root(
        ctx,
        ["ROCM_PATH", "HIP_PATH"],
        ["/opt/rocm"],
        [("/opt", "rocm-")],
        _valid_rocm_root,
    )

    build = [
        'load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")',
        'package(default_visibility = ["//visibility:public"])',
    ]

    if root:
        runtime = _find_rocm_runtime_library(ctx, root)

        if not runtime:
            fail("Found ROCm toolkit at %s but could not locate libamdhip64" % root)

        _symlink_entries(ctx, root, ["bin", "include", "lib", "lib64"])

        build.extend([
            'filegroup(name = "hipcc", srcs = ["bin/hipcc"])',
            'cc_library(',
            '    name = "rocm_sdk_core",',
            '    srcs = [],',
            '    hdrs = [],',
            '    includes = ["include"],',
            ')',
            'cc_import(',
            '    name = "amdhip64",',
            '    shared_library = "%s",' % runtime[len(root) + 1:],
            ')',
        ])
    else:
        build.extend([
            'filegroup(name = "hipcc", srcs = [])',
            'cc_library(name = "rocm_sdk_core", srcs = [], hdrs = [], includes = [])',
            'cc_library(name = "amdhip64", srcs = [], hdrs = [])',
        ])

    ctx.file("WORKSPACE", 'workspace(name = "%s")\n' % ctx.name)
    ctx.file("BUILD.bazel", "\n".join(build) + "\n")

def _discover_libpython_path(ctx):
    python_bin = ctx.os.environ.get("PYTHON_BIN_PATH")

    if not python_bin:
        conda_prefix = ctx.os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidate = conda_prefix + "/bin/python3"
            if _path_exists(ctx, candidate):
                python_bin = candidate

    if not python_bin:
        for candidate in ["/usr/bin/python3", "/usr/local/bin/python3"]:
            if _path_exists(ctx, candidate):
                python_bin = candidate
                break

    if not python_bin:
        result = ctx.execute(
            ["/usr/bin/env", "python3", "-c", "import sys; print(sys.executable)"],
            quiet = True,
        )

        if result.return_code == 0:
            python_bin = result.stdout.strip()

    if not python_bin:
        fail("Unable to locate python3 for libpython discovery. Set PYTHON_BIN_PATH or CONDA_PREFIX.")

    probe = """
import glob
import os
import sys
import sysconfig

libdir = sysconfig.get_config_var("LIBDIR") or ""
libpl = sysconfig.get_config_var("LIBPL") or ""
search_roots = []
names = []
shared = []
static = []

for value in [sysconfig.get_config_var("LDLIBRARY"), sysconfig.get_config_var("LIBRARY")]:
    if value and value not in names:
        names.append(value)

for candidate in [libdir, libpl, os.path.join(sys.prefix, "lib")]:
    if candidate and candidate not in search_roots and os.path.isdir(candidate):
        search_roots.append(candidate)

for root in search_roots:
    for name in names:
        path = os.path.join(root, name)
        if name.endswith(".a"):
            static.append(path)
        else:
            shared.append(path)
    for pattern in ["libpython*.so", "libpython*.so.*"]:
        shared.extend(sorted(glob.glob(os.path.join(root, pattern))))
    static.extend(sorted(glob.glob(os.path.join(root, "libpython*.a"))))

candidates = shared + static
seen = set()
for candidate in candidates:
    if candidate in seen:
        continue
    seen.add(candidate)
    if os.path.exists(candidate):
        print(candidate)
        raise SystemExit(0)

raise SystemExit(1)
"""

    result = ctx.execute([python_bin, "-c", probe], quiet = True)

    if result.return_code != 0:
        fail("Unable to locate libpython using %s" % python_bin)

    libpython = result.stdout.strip()

    if not libpython:
        fail("Python discovery did not return a libpython path.")

    return libpython

def _local_libpython_repo_impl(ctx):
    libpython = _discover_libpython_path(ctx)
    libpython_name = _basename(libpython)
    import_rule = 'cc_import(name = "libpython_import", shared_library = "%s")' % libpython_name

    if libpython_name.endswith(".a"):
        import_rule = 'cc_import(name = "libpython_import", static_library = "%s")' % libpython_name

    ctx.symlink(ctx.path(libpython), libpython_name)

    build = [
        'load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")',
        'package(default_visibility = ["//visibility:public"])',
        import_rule,
        'cc_library(',
        '    name = "libpython",',
        '    linkopts = [',
        '        "-lpthread",',
        '        "-lrt",',
        '        "-ldl",',
        '        "-lutil",',
        '        "-lcrypt",',
        '        "-lm",',
        '    ],',
        '    deps = [',
        '        ":libpython_import",',
        '        "@local_config_python//:python_headers",',
        '    ],',
        '    linkstatic = 1,',
        ')',
    ]

    ctx.file("WORKSPACE", 'workspace(name = "%s")\n' % ctx.name)
    ctx.file("BUILD.bazel", "\n".join(build) + "\n")

local_cuda_sdk_repository = repository_rule(
    implementation = _local_cuda_sdk_repo_impl,
    environ = ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"],
)

local_rocm_sdk_repository = repository_rule(
    implementation = _local_rocm_sdk_repo_impl,
    environ = ["HIP_PATH", "ROCM_PATH"],
)

local_libpython_repository = repository_rule(
    implementation = _local_libpython_repo_impl,
    environ = ["CONDA_PREFIX", "PYTHON_BIN_PATH"],
)
