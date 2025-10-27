import os
import numpy as np
import matplotlib.pyplot as plt
import optuna

from scipy.interpolate import make_smoothing_spline

from xasproc.io import read_feff_xmu_file
from xasproc.xas_spectrum import XASSpectrum
from xasproc.xas_collection import XASCollection
from xasproc.preprocess import scale, interpolate, integrate

from boxas.exp_data import load_spectrum
from boxas.load_xmu import load_best_trial_spectrum

figsize = (6,5)

metric_names = {
    'l2_norm_dist': 'L2 Normalized Distance',
    'r_factor': 'R-Factor',
    'cosine_sim': 'Cosine Similarity',
    'pearson_corr': 'Pearson Correlation',
    'spearman_corr': 'Spearman Correlation'
}

def plot_measured_spectra(cfg, materials, e_range):
    m0 = materials[0]
    cfg0 = cfg['exp_data'][m0]
    F = load_spectrum(cfg0, m0, e_range=e_range)
    if m0.startswith('Pd'):
        F = None

    # Create figure with two subplots
    fig, ax = plt.subplots(
        figsize=(5,4), 
    )
    
    # Get measured material spectrum
    for m in materials:
        cfgm = cfg['exp_data'][m]
        M = load_spectrum(cfgm, m, foil=F, e_range=e_range)

        ax.plot(M.energy, M.mu, label=cfgm['label'], linewidth=2)

    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Normalized μ(E) [a.u.]')
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(e_range)

    return fig


def plot_optimized_spectra(cfg, tool, material, metrics, e_range):
    """
    Plot optimized spectra vs measured spectrum
    
    Args:
        cfg: Configuration dictionary
        material: Material name
        metrics: List of metrics used for optimization
        e_range: Tuple of (e_min, e_max) for plotting
    """
    
    F = None
    cfgm = cfg['exp_data'][material]
    e_min, e_max = e_range
    norm_e_min, norm_e_max = cfgm['energy']['min'], cfgm['energy']['max']

    # Get measured material spectrum
    M = load_spectrum(cfgm, material, foil=F, e_range=e_range)
    m_integral = integrate(M, norm_e_min, norm_e_max)[0]
    
    # Storage for spectra and their names
    spectra = [M]
    names = ['Measured']
    colors = ['black']
    
    # Storage for the database connection
    storage = "mysql+pymysql://optuna:optuna_pw@localhost/optuna_db"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, 
        gridspec_kw={'height_ratios': [2, 1]},
        figsize=figsize, 
        sharex=True,
        clear=True
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Plot measured spectrum in the top subplot
    ax1.plot(M.energy, M.mu, 'k-', label=f'{cfgm["label"]} Exp.', linewidth=2)
    
    # Define a color cycle for the metrics
    metric_colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    
    for i, metric in enumerate(metrics):
        study_name = f"{tool}_{material}_{metric}"
        
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        # Get the best trial
        best_trial = study.best_trial
        X = load_best_trial_spectrum(cfgm, tool=tool, study_name=study_name)
        
        # integral normalize the spectrum
        C = XASCollection([X], njobs=0)
        # metric_integral = C.pipe(integrate, e_min, e_max)[0][0]
        metric_integral = integrate(X, norm_e_min, norm_e_max)[0]
        energy_mask = np.logical_and(M.energy >= e_min, M.energy <= e_max)
        X_norm = (
            C
            .pipe(scale, [m_integral/metric_integral], zip_args=True)
            .pipe(interpolate, M.energy)
        )[0]

        # Add to the list of spectra
        spectra.append(X_norm)
        names.append(f"{metric} (Trial {best_trial.number})")
        colors.append(metric_colors[i])
        
        # Plot in the top subplot
        metric_name = f"{metric_names[metric]}" if material == 'Ni_aps' else None
        ax1.plot(X_norm.energy, X_norm.mu, color=metric_colors[i], 
                    label=metric_name, linewidth=1.5)
        
        interp_coll = XASCollection([X_norm, M]).pipe(interpolate, M.energy)
        X_interp, M_interp = interp_coll[0], interp_coll[1]
        
        # Calculate ratio and plot only in the normalization range
        mask = (X_interp.energy >= norm_e_min) & (X_interp.energy <= norm_e_max)
        delta = 100*(X_interp.mu[mask] - M_interp.mu[mask]) / M_interp.mu[mask]
        ax2.plot(X_interp.energy[mask], delta, color=metric_colors[i], 
                    label=f"{metric} delta", linewidth=1.5)
            
    # Add horizontal line at ratio=1 in the bottom subplot
    ax1.axvline(x=X_interp.energy[mask].min(), color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=X_interp.energy[mask].max(), color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=X_interp.energy[mask].min(), color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=X_interp.energy[mask].max(), color='k', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
    
    # Set labels and titles
    # ax1.set_title(f"{material} XAS Spectra Comparison", fontsize=14)
    # ax1.set_xlabel("Energy (eV)", fontsize=12)
    ax1.set_ylabel("Normalized μ(E) [a.u.]", fontsize=12)
    ax1.set_ylim(0, 1.2)
    ax1.legend(loc='lower right', fontsize=10)
    # ax1.grid(True, alpha=0.3)
    
    # ax2.set_title("Ratio to Measured Spectrum", fontsize=14)
    ax2.set_xlabel("Energy [eV]", fontsize=12)
    ax2.set_ylabel("$\Delta$ [%]", fontsize=12)
    ax2.set_xlim(e_min, e_max)
    ax2.set_ylim(-10, 9)
    # ax2.grid(True, alpha=0.3)
    
    # Add text with normalization range
    # fig.text(0.01, 0.01, f"L2 normalization range: [{norm_e_min}, {norm_e_max}] eV", 
    #          fontsize=8, ha='left', va='bottom')
            
    return fig, spectra, names

def read_compound_spectrum(cfg_dir, M, cfg_compounds, study_name, energy_offsets, best_trial):
    # Get the run directory for the best trial
    best_params = best_trial.params
    print(best_params)

    X_out = XASSpectrum(
        energy=M.energy,
        mu=np.array([0.]*len(M.energy)),
    )

    fcb_mix = [best_params['fcc_bcc_mixture'], 1.-best_params['fcc_bcc_mixture']]

    for ic, an in enumerate(cfg_compounds):
        # Read the FEFF output
        run_dir = os.path.join(cfg_dir['project_root'], 'runs', study_name, f'run_{an}_{best_trial.number:03d}')
    
        xmu_file = os.path.join(run_dir, 'xmu.dat')
        if not os.path.exists(xmu_file):
            print(f"Warning: xmu.dat not found for {study_name} best trial {best_trial.number}")
            continue
            
        xmu_info, D = read_feff_xmu_file(xmu_file)
        
        # Apply energy offset
        D['omega'] = D['omega'] - energy_offsets[ic]
        
        # Create spectrum from FEFF output
        X_out.mu += fcb_mix[ic]*make_smoothing_spline(D['omega'], D['mu'], lam=0.01)(M.energy)
    
    return X_out
        
def plot_compound_spectra(cfg, material, metrics, e_range):
    """
    Plot optimized spectra vs measured spectrum
    
    Args:
        cfg: Configuration dictionary
        material: Material name
        metrics: List of metrics used for optimization
        e_range: Tuple of (e_min, e_max) for plotting
    """
    
    if material.startswith('Fe2Ni8'):
        F = load_spectrum(cfg['exp_data']['Ni'], 'Ni', e_range=(8100, 8900))
    else:
        F = None
    
    cfg_dir = cfg['dir']
    cfg = cfg['exp_data'][material]
    cfg_compounds = cfg['compounds']
    e_min, e_max = e_range
    norm_e_min, norm_e_max = cfg['energy']['min'], cfg['energy']['max']

    # Get measured material spectrum
    M = load_spectrum(cfg, material, foil=F, e_range=e_range)
    mu_norm = np.linalg.norm(M.mu[(M.energy >= norm_e_min) & (M.energy <= norm_e_max)])
    
    # Storage for spectra and their names
    spectra = [M]
    names = ['Measured']
    colors = ['black']
    
    # Storage for the database connection
    storage = "mysql+pymysql://optuna:optuna_pw@localhost/optuna_db"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, 
        gridspec_kw={'height_ratios': [2, 1]},
        figsize=figsize, 
        sharex=True
    )
    
    # Plot measured spectrum in the top subplot
    ax1.plot(M.energy, M.mu, 'k-', label=f'{cfg["label"]} Exp.', linewidth=2)
    
    # Define a color cycle for the metrics
    metric_colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    
    for i, metric in enumerate(metrics):
        study_name = f"{material}_{metric}"
        
        # try:
        # Load the study from the database
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        # Get the best trial
        best_trial = study.best_trial
        
        energy_offsets = [eo for ee, eo in best_trial.user_attrs.get("custom_info", {}).items() if ee.startswith('energy_offset_')]
        X_norm = read_compound_spectrum(cfg_dir, M, cfg_compounds, study_name, energy_offsets, best_trial)
        
        # Add to the list of spectra
        spectra.append(X_norm)
        names.append(f"{metric} (Trial {best_trial.number})")
        colors.append(metric_colors[i])
        
        # Plot in the top subplot
        metric_name = metric_names[metric]
        # ax1.plot(X_norm.energy, 1.09*(X_norm.mu-0.03), '-', color=metric_colors[i], 
        ax1.plot(X_norm.energy+4.3, X_norm.mu, '-', color=metric_colors[i], 
                    label=f"{metric_name}", linewidth=1.5)
        
        # Calculate and plot the ratio in the bottom subplot
        # First, ensure both spectra are on the same energy grid
        interp_coll = XASCollection([X_norm, M]).pipe(interpolate, M.energy)
        X_interp, M_interp = interp_coll[0], interp_coll[1]
        
        # Calculate ratio and plot only in the normalization range
        mask = (X_interp.energy >= norm_e_min) & (X_interp.energy <= norm_e_max)
        delta = 100*(X_interp.mu[mask] - M_interp.mu[mask]) / M_interp.mu[mask]
        ax2.plot(X_interp.energy[mask], delta, '-', color=metric_colors[i], 
                    label=f"{metric} delta", linewidth=1.5)
            
        # except Exception as e:
        #     print(f"Error processing {study_name}: {e}")
    
    # Add horizontal line at ratio=1 in the bottom subplot
    ax1.axvline(x=X_interp.energy[mask].min(), color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=X_interp.energy[mask].max(), color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=X_interp.energy[mask].min(), color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=X_interp.energy[mask].max(), color='k', linestyle='--', alpha=0.5)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    # Set labels and titles
    # ax1.set_title(f"{material} XAS Spectra Comparison", fontsize=14)
    # ax1.set_xlabel("Energy (eV)", fontsize=12)
    ax1.set_ylabel("Normalized μ(E) [a.u.]", fontsize=12)
    # ax1.set_xlim(e_min, e_max)
    ax1.legend(loc='best', fontsize=10)
    # ax1.grid(True, alpha=0.3)
    
    # ax2.set_title("Ratio to Measured Spectrum", fontsize=14)
    ax2.set_xlabel("Energy [eV]", fontsize=12)
    ax2.set_ylabel("$\Delta$ [%]", fontsize=12)
    ax2.set_xlim(e_min, e_max)
    ax2.set_ylim(-10, 10)
    # ax2.grid(True, alpha=0.3)
    
    # Add text with normalization range
    # fig.text(0.01, 0.01, f"L2 normalization range: [{norm_e_min}, {norm_e_max}] eV", 
    #          fontsize=8, ha='left', va='bottom')
            
    plt.subplots_adjust(hspace=0)
    
    plt.tight_layout()
    
    return fig, spectra, names