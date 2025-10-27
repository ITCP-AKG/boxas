import subprocess
import os
import shutil
from time import sleep

from xasproc.xas_spectrum import XASSpectrum

from boxas.config import get_config, TOOL_CONFIGS


def runner(tool, run, template, compounds, parameters, trial_index):
    # Create the trial directory
    cfg = get_config('cfg_foils.yaml')
    project_root = cfg['dir']['project_root']
    run_dir = cfg['dir']['run_dir']
    template_dir = project_root + '/' + cfg['dir']['template_dir']

    executable = cfg['exec'][tool]
    tool_cfg = TOOL_CONFIGS[tool]

    rendered_header = template.render(parameters)

    R = []

    for an, alloy in compounds.items():
        c_factor = parameters.get('c_factor', None)
        if len(alloy['elements']) == 1:
            c_factor = None

        potatoms = tool_cfg.make_atoms(alloy, parameters.get('alloy_fraction'), c_factor)

        trial_dir = f'{project_root}/{tool}_{run_dir}/{run}/run_{an}_{trial_index:03d}'
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)

        sleep(1)
        os.makedirs(trial_dir, exist_ok=True)
        # Write the rendered input to feff.inp in the trial directory
        # Run FEFF in the trial directory using subprocess
        sleep(trial_index%3 )  # add some delay to start feff in different times
        
        if tool == 'fdmnes':
            # copy fdmfile to trial directory
            shutil.copy(template_dir + '/fdmfile.txt', trial_dir)

        try:
            # Change to the trial directory
            os.chdir(trial_dir)
            
            with open(os.path.join(trial_dir, tool_cfg.inp_fname), 'w') as f:
                f.write(rendered_header+potatoms)

            # Run FEFF
            _ = subprocess.run([executable], capture_output=True, text=True, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running {an} {tool} for trial {trial_index}")

            return e

        output_file = os.path.join(trial_dir, tool_cfg.out_fname)
        try:
            xmu_info, D = tool_cfg.reader(output_file)
        except FileNotFoundError:
            print(f"{tool_cfg.out_fname} not found for trial {trial_index}")
            return None

        X = XASSpectrum().from_feff(xmu_info, D, energy_column=tool_cfg.energy_col, mu_expression='mu')
        R.append(X)

    if len(R) > 1:
        R[0].mu *= parameters['fcc_bcc_mixture']
        R[1].mu *= (1.-parameters['fcc_bcc_mixture'])

    return R
