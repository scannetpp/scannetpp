import subprocess
import sys
import yaml
import json
from munch import Munch


def read_txt_list(path):
    with open(path) as f:
        lines = f.read().splitlines()

    return lines


def load_json(path):
    with open(path) as f:
        j = json.load(f)

    return j


def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)


def run_command(cmd: str, verbose=False, exit_on_error=True):
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        if out.stderr is not None:
            print(out.stderr.decode("utf-8"))
        if exit_on_error:
            sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out
