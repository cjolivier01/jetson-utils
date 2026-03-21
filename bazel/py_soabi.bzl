def _py_soabi_impl(repository_ctx):
    # Prefer Python from CONDA_PREFIX if available, else fall back to system python3
    conda_prefix = repository_ctx.os.environ.get("CONDA_PREFIX")
    python_bin = (conda_prefix + "/bin/python3") if conda_prefix else "python3"

    # Execute Python to get the SOABI value.
    result = repository_ctx.execute(
        [python_bin, "-c", "import sysconfig; print(sysconfig.get_config_var('SOABI'))"],
        quiet = True,
    )
    if result.return_code != 0:
        fail("Failed to run python command: " + result.stderr)
    soabi = result.stdout.strip()
    
    # Write the soabi.bzl file containing the SOABI variable.
    repository_ctx.file("soabi.bzl", "SOABI = \"%s\"\n" % soabi)
    
    # Also create an empty BUILD file so that the repository is a valid package.
    repository_ctx.file("BUILD", "")

py_soabi = repository_rule(
    implementation = _py_soabi_impl,
    local = True,
    environ = ["CONDA_PREFIX"],
)
