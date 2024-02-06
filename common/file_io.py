

import yaml
from munch import Munch
import json
import yaml

def write_json(path, data):
    with open(path, "w") as f:
        f.write(json.dumps(data, indent=4))

def load_json(path):
    with open(path) as f:
        j = json.load(f)

    return j

def load_yaml(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return y


def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()

    return lines

def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)

def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)