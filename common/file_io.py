

import yaml
from munch import Munch

def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()

    return lines

def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)