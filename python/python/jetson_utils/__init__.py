
# print("jetson_utils.__init__.py")

# Prefer the in-package extension (wheel install),
# but also support Bazel/dev runfiles by falling back to top-level or dynamic load.
try:
    from .jetson_utils_python import *  # type: ignore
except Exception:
    try:
        from jetson_utils_python import *  # type: ignore
    except Exception:
        # Last resort: dynamically locate the extension in runfiles/sys.path
        import sys, glob, importlib.util, importlib.machinery, os
        _loaded = False
        # Also scan upwards from this file for typical Bazel runfiles layout
        _parents = []
        _d = os.path.abspath(__file__)
        for _ in range(8):
            _d = os.path.dirname(_d)
            _parents.append(_d)

        for p in list(dict.fromkeys(sys.path + [os.path.dirname(__file__)] + _parents)):
            try:
                # search both directly under p and recursively
                candidates = glob.glob(os.path.join(p, "jetson_utils_python.*.so"))
                candidates += glob.glob(os.path.join(p, "**", "jetson_utils_python.*.so"), recursive=True)
                for so in candidates:
                    # Ensure core shared lib is preloaded for unresolved symbols
                    try:
                        import ctypes
                        so_dir = os.path.dirname(so)
                        # preload libjetson_cuda then libjetson_utils for symbol resolution
                        preload_libs = [
                            ("libjetson_cuda.so", [
                                os.path.join(so_dir, "libjetson_cuda.so"),
                                os.path.join(so_dir, "..", "..", "python", "extdeps", "libjetson_cuda.so"),
                            ]),
                            ("libjetson_utils.so", [
                                os.path.join(so_dir, "libjetson_utils.so"),
                                os.path.join(so_dir, "..", "..", "core", "libjetson_utils.so"),
                            ]),
                        ]

                        for libname, defaults in preload_libs:
                            for cc in defaults:
                                if os.path.exists(cc):
                                    ctypes.CDLL(os.path.realpath(cc), mode=ctypes.RTLD_GLOBAL)
                                    break
                            else:
                                # scan upwards for it
                                for parent in _parents:
                                    cc = os.path.join(parent, libname)
                                    if os.path.exists(cc):
                                        ctypes.CDLL(os.path.realpath(cc), mode=ctypes.RTLD_GLOBAL)
                                        break
                    except Exception:
                        pass

                    spec = importlib.util.spec_from_file_location("jetson_utils_python", so)
                    if spec and isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                        globals().update(mod.__dict__)
                        _loaded = True
                        break
                if _loaded:
                    break
            except Exception:
                continue
        if not _loaded:
            raise

VERSION = '1.0.1'
