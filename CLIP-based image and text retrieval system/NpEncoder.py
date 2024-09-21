import json
from datetime import time

import chardet
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        if isinstance(obj, bytes):
            print(obj)
            return str(obj)
        else:
            return super(NpEncoder, self).default(obj)
