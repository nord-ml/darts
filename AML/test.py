from utils import fix_pythonpath_if_working_locally
fix_pythonpath_if_working_locally()

import darts
import sys

print(darts.__file__)
print(sys.path)

from darts.models import TFTSSMModel



