__version__ = "1.2.0"

import os
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category = UserWarning)