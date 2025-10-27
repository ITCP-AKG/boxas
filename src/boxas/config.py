import yaml
from pathlib import Path
from collections import namedtuple

from xasproc.io import read_feff_xmu_file, read_fdmnes_xmu_file
from boxas.make_atoms import make_atoms_feff, make_crystal_fdmnes

ToolConfig = namedtuple('ToolConfig', ['inp_fname', 'out_fname', 'reader', 'energy_col', 'make_atoms'])

TOOL_CONFIGS = {
    'feff': ToolConfig(
        inp_fname='feff.inp',
        out_fname='xmu.dat',
        reader=read_feff_xmu_file,
        energy_col='omega',
        make_atoms=make_atoms_feff,
    ),
    'fdmnes': ToolConfig(
        inp_fname='fdmnes_inp.txt',
        out_fname='fdmnes_out_conv.txt',
        reader=read_fdmnes_xmu_file,
        energy_col='energy',
        make_atoms=make_crystal_fdmnes,
    )
}

def get_config(cfg_file="cfg_foils.yaml"):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / cfg_file).exists():
            with open(parent / cfg_file) as f:
                _config = yaml.safe_load(f)
                _config["dir"]["project_root"] = str(parent)

                return _config

    raise FileNotFoundError(f"{cfg_file} not found")