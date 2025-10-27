import os

import optuna

from xasproc.preprocess import get_edge
from xasproc.xas_spectrum import XASSpectrum

from boxas.config import TOOL_CONFIGS, get_config


def load_simulated_spectrum(tool, xmu_file):

    tool_cfg = TOOL_CONFIGS[tool]
    
    xmu_info, D = tool_cfg['reader'](xmu_file)

    # get preliminary spectrum
    X = XASSpectrum().from_feff(xmu_info, D, energy_column=tool_cfg.energy_col, mu_expression='mu')

    return X


def load_best_trial_spectrum(cfgm, tool, study_name):

    storage = "mysql+pymysql://optuna:optuna_pw@localhost/optuna_db"
    tool_cfg = TOOL_CONFIGS[tool]

    study = optuna.load_study(study_name=study_name, storage=storage)

    cfg_dir = get_config()['dir']

    # Get the best trial
    best_trial = study.best_trial

    # Get the run directory for the best trial
    an = list(cfgm['compounds'].keys())[0]
    run_dir = os.path.join(cfg_dir['project_root'], f'{tool}_runs', study_name, f'run_{an}_{best_trial.number:03d}')

    # Read the FEFF output
    xmu_file = os.path.join(run_dir, tool_cfg.out_fname)
    # print(xmu_file)
    if not os.path.exists(xmu_file):
        print(f"Warning: {tool_cfg.out_fname} not found for {study_name} best trial {best_trial.number}")
        return None
    
    xmu_info, D = tool_cfg.reader(xmu_file)

    # Get energy offset from the database if available
    energy_scales = [eo for ee, eo in best_trial.user_attrs.get("custom_info", {}).items() if ee.startswith('energy_scale_')]
    energy_scale = energy_scales[0] if energy_scales else 0.0

    # get preliminary spectrum
    X = XASSpectrum().from_feff(xmu_info, D, energy_column=tool_cfg.energy_col, mu_expression='mu')

    xe0 = get_edge(X)
    e0 = cfgm['energy']['edge']
    D[tool_cfg.energy_col] = e0 + (D[tool_cfg.energy_col] - xe0)/energy_scale
    X_aligned = XASSpectrum().from_feff(xmu_info, D, energy_column=tool_cfg.energy_col, mu_expression='mu')

    return X_aligned
        
