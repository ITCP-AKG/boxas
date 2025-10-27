import numpy as np
import optuna
from xasproc.xas_collection import XASCollection
from xasproc.preprocess import interpolate, integrate, scale
from xasproc.metrics import stat_scores

from boxas.exp_data import load_spectrum
from boxas.load_xmu import load_best_trial_spectrum

metric_names = {
    'l2_norm_dist': 'L2 Normalized Distance',
    'r_factor': 'R-Factor',
    'cosine_sim': 'Cosine Similarity',
    'pearson_corr': 'Pearson Correlation',
    'spearman_corr': 'Spearman Correlation'
}

opt_params_names = {
    'Vr': '$V_r$',
    'Vi': '$V_i$',
    'r_scf': '$R_{{SCF}}$',
    'r_fms': '$R_{{FMS}}$'
}

latex_header = {
    'Vr': 'p{{1cm}}',
    'Vi': 'p{{1cm}}',
    'r_scf': 'r',
    'r_fms': 'r'
}

def build_parameters_table(cfg, materials, metrics, opt_params):
    """
    Compose optimization results table in LaTeX format
    
    Args:
        cfg: Configuration dictionary
        material: Material name
        metrics: List of metrics to provide results for
        opt_params: List of parameters the optimization was performed on
    """
    
    # Storage for the database connection
    storage = "mysql+pymysql://optuna:optuna_pw@localhost/optuna_db"
    
    tables = """\\begin{{table}}
\\centering
\\begin{{tabular}}{{lll""" + ''.join([latex_header[op] for op in opt_params]) + """}}
\\midrule
 Objective & Sample & Value & """ + ' & '.join([opt_params_names[op] for op in opt_params]) + """ \\\\
{optimized_pars}\\midrule
\\end{{tabular}}
\\caption{{ FEFF parameters tuned to simulate given sample using different objective functions.\\label{{tab:params_feff}}}}
\\end{{table}}
"""
    optimized_pars = ''
    
    for i, metric in enumerate(metrics):
        optimized_pars += f"\\midrule \n{metric_names[metric]}"

        for material in materials:
            cfgm = cfg['exp_data'][material]
            
            # Get measured material spectrum
            study_name = f"{material}_{metric}"
            
            # Load the study from the database
            study = optuna.load_study(study_name=study_name, storage=storage)
            
            # Get the best trial
            best_trial = study.best_trial
            best_params = best_trial.params
            
            optimized_pars += f"""& {cfgm['label']} & {best_trial.value:.2g}  & """ \
                + '& '.join([f"{best_params[op]:.2g}" for op in opt_params]) + """ \\\\ \n"""

    filled_table = tables.format(tex_label=f'tab:{material}', optimized_pars=optimized_pars)
    
    return filled_table


def build_rfactor_table(cfg, materials, metrics):
    """
    Compose optimization results table in LaTeX format
    
    Args:
        cfg: Configuration dictionary
        materias: List of materials
        metrics: List of metrics to provide results for
    """
    
    tables = """\\begin{{table}}
\\centering
\\begin{{tabular}}{{lrr""" + ''.join(['r' for m in materials]) + """}}
\\midrule
 Objective & """ + ' & '.join([cfg["exp_data"][m]['label'] for m in materials]) + """ \\\\
\\midrule
{r_factors}\\midrule
\\end{{tabular}}
\\caption{{ R-factors calculated for FEFF spectra optimized to different samples using selected metrics.\\label{{tab:feff_rfactor}}}}
\\end{{table}}
"""
    r_factors = ''
    
    for i, metric in enumerate(metrics):
        r_factors += f"""{metric_names[metric]} & """

        for i, material in enumerate(materials):
            cfgm = cfg['exp_data'][material]
            e_min, e_max = cfgm['energy']['min'], cfgm['energy']['max']
            e_range = (e_min-15, e_max+20)
            
            # Get measured material spectrum
            M = load_spectrum(cfgm, material, e_range=e_range)
            m_integral = integrate(M, e_min, e_max)[0]

            # Get the best trial spectrum
            study_name = f"{material}_{metric}"
            X = load_best_trial_spectrum(cfgm, tool='feff', study_name=study_name)

            # integral normalize the spectrum
            C = XASCollection([X], njobs=0)
            metric_integral = C.pipe(integrate, e_min, e_max)[0][0]
            print(metric, m_integral/metric_integral)
            energy_mask = np.logical_and(M.energy >= e_min, M.energy <= e_max)
            X_norm = (
                C.pipe(scale, [m_integral/metric_integral], zip_args=True)
                .pipe(interpolate, M.energy)
            )[0]

            # Calculate r-factor
            r_factor = stat_scores['r_factor'](M.mu[energy_mask], X_norm.mu[energy_mask])

            r_factors += f""" {r_factor:.2g} & """

        r_factors += '\\\\ \n'

    filled_table = tables.format(r_factors=r_factors)
    
    return filled_table