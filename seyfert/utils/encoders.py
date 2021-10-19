import numpy as np
import json
from typing import Dict


class PhysParJSONEncoder(json.JSONEncoder):
    def default(self, o: "object") -> "Dict":
        return o.__dict__


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
