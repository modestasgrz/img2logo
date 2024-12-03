import json
from typing import Dict


class Config(Dict):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as f:
            data = json.loads(f.read())

        self.update(data)
