
# print("jetson.utils.__init__.py")

# Back-compat shim that re-exports from jetson_utils
from jetson_utils import *  # type: ignore

VERSION = '1.0.1'

print("warning:  importing jetson.utils is deprecated.  please 'import jetson_utils' instead.")
