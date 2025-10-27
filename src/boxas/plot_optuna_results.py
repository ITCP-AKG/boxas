import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
import optuna
from optuna.importance import get_param_importances

import seaborn as sns
from scipy.stats import pearsonr

metric_names = {
    'l2_norm_dist': 'L2 Normalized Distance',
    'r_factor': 'R-Factor',
    'cosine_sim': 'Cosine Similarity',
    'pearson_corr': 'Pearson Correlation',
    'spearman_corr': 'Spearman Correlation'
}

def plot_contours(cfg, material, params, metric, tag=None):
    """
    Plot contours of the optimization landscape for different parameters across metrics.

    Args:
        cfg: Configuration dictionary
        material: Material name
        metrics: List of metrics used for optimization
    """
    storage = "mysql+pymysql://optuna:optuna_pw@localhost/optuna_db"
    
    # Create a figure with subplots for each parameter pair
    n_params = len(params)
    fig, axes = plt.subplots(n_params-1, n_params-1, figsize=(2*n_params, 2*n_params), squeeze=False)

    ngrid = 100
    minz = 100.
    maxz = -100.

    study_name = f"{material}_{metric}_{tag}" if tag else f"{material}_{metric}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    for i in range(n_params-1):
        for j in range(i+1, n_params):
            ax = axes[i][j-1]
            param1, param2 = params[i], params[j]
            print(i, param1, j, param2)
            
            # Extract parameter values and objective values
            x = [t.params[param1] for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            y = [t.params[param2] for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            z = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

            minz = min(minz, min(z))
            maxz = max(maxz, max(z))

    # Create a normalize object for consistent color scaling
    norm = Normalize(vmin=minz, vmax=maxz)

    for i in range(n_params-1):
        for j in range(i+1, n_params):
            ax = axes[i][j-1]
            param1, param2 = params[i], params[j]
            
            # Extract parameter values and objective values
            x = [t.params[param1] for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            y = [t.params[param2] for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            z = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

            # Create grid values first.
            xi = np.linspace(min(x), max(x), ngrid)
            yi = np.linspace(min(y), max(y), ngrid)

            # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
            triang = tri.Triangulation(x, y)
            interpolator = tri.LinearTriInterpolator(triang, z)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)

            contour = ax.contourf(xi, yi, zi, levels=20, cmap='GnBu_r', norm=norm)
            
            # Set labels and title
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            # ax.set_xlim(cfg['feff_params']['r_scf']['bounds'])
            # ax.set_ylim(cfg['feff_params']['r_fms']['bounds'])
            # ax.set_title(f"{metric_names[metric]}")

            # Plot the best point
            best_params = study.best_params
            ax.plot(best_params[param1], best_params[param2], 'r*', markersize=10, label='Best parameters')
            
            # Add legend
            # ax.legend()

    # Add a colorbar to the figure
    cbar_ax = fig.add_axes([0.15, 0.1, 0.03, 0.4], )  # [left, bottom, width, height]
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_label(metric_names[metric], rotation=90, va='bottom')

    # Remove empty subplots
    for i in range(n_params-1):
        for j in range(n_params-1):
            if j < i:
                fig.delaxes(axes[i][j])

    plt.tight_layout()

    return fig

def plot_rep_contours(cfg, material, params, metric, tags):
    """
    Plot contours of the optimization landscape for different parameters across metrics.

    Args:
        cfg: Configuration dictionary
        material: Material name
        metrics: List of metrics used for optimization
    """
    storage = "mysql+pymysql://optuna:optuna_pw@localhost/optuna_db"
    
    # Create a figure with subplots for each parameter pair
    fig, axes = plt.subplots(3, 3, figsize=(8,8), squeeze=False)

    ngrid = 100
    minz = 100.
    maxz = -100.

    for tag in tags:
        study_name = f"{material}_{metric}_{tag}"
        study = optuna.load_study(study_name=study_name, storage=storage)
    
        param1, param2 = params[0], params[1]
        
        # Extract parameter values and objective values
        x = [t.params[param1] for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        y = [t.params[param2] for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        z = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        minz = min(minz, min(z))
        maxz = max(maxz, max(z))

    # Create a normalize object for consistent color scaling
    norm = Normalize(vmin=minz, vmax=maxz)

    for itag, tag in enumerate(tags):
        study_name = f"{material}_{metric}_{tag}"
        study = optuna.load_study(study_name=study_name, storage=storage)
    
        param1, param2 = params[0], params[1]
        
        ax = axes[itag//3][itag%3]
        
        # Extract parameter values and objective values
        x = [t.params[param1] for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        y = [t.params[param2] for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        z = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        # Create grid values first.
        xi = np.linspace(min(x), max(x), ngrid)
        yi = np.linspace(min(y), max(y), ngrid)

        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        contour = ax.contourf(xi, yi, zi, levels=20, cmap='GnBu_r', norm=norm)
        
        # Set labels and title
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_xlim(cfg['feff_params']['Vi']['bounds'])
        ax.set_ylim(cfg['feff_params']['r_fms']['bounds'])

        # Plot the best point
        best_params = study.best_params
        ax.plot(best_params[param1], best_params[param2], 'r*', markersize=10, label='Best parameters')
        
        # Add legend
        # ax.legend()

    # Add a colorbar to the figure
    cbar_ax = fig.add_axes([1.02, 0.075, 0.03, 0.9], )  # [left, bottom, width, height]
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_label(metric_names[metric], rotation=90, va='bottom')

    plt.tight_layout()

    return fig



def plot_optuna_pi(cfg, material, params, metrics):
    storage = "mysql+pymysql://optuna:optuna_pw@localhost/optuna_db"
    
    pi = {}

    for met in metrics:
        study_name = f"{material}_{met}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        pi[met] = get_param_importances(study)

    D = pd.DataFrame(pi)[metrics]
    D.rename(columns=metric_names, inplace=True)

    ax = D.T[params].plot.barh(stacked=True, figsize=(7, 4), alpha=0.8)
    ax.set_xlabel("Parameter importance", fontsize=12)

    # Place the legend above the plot, spread horizontally
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=len(params), frameon=False)
    
    fig = ax.get_figure()

    return fig

def plot_pairwise_correlations(cfg, material, params, metric):
    storage = "mysql+pymysql://optuna:optuna_pw@localhost/optuna_db"
    
    study_name = f"{material}_{metric}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    # Extract parameter values
    param_values = {param: [] for param in params}
    values = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            for param in params:
                param_values[param].append(trial.params[param])
            values.append(trial.value)
    
    # Calculate correlation matrix
    corr_matrix = np.zeros((len(params), len(params)))
    for i, param1 in enumerate(params):
        for j, param2 in enumerate(params):
            corr, _ = pearsonr(param_values[param1], param_values[param2])
            corr_matrix[i, j] = corr

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1, center=0,
                xticklabels=params, yticklabels=params, ax=ax)

    # Customize the plot
    plt.tight_layout()

    # Adjust font sizes
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    ax.set_title(ax.get_title(), fontsize=14)

    # Customize the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Correlation', fontsize=12)

    return fig

def save_contour_plots(cfg, material, metrics, output_dir=None):
    """
    Generate and save contour plots for the optimization landscape.

    Args:
        cfg: Configuration dictionary
        material: Material name
        metrics: List of metrics used for optimization
        output_dir: Directory to save the figure. If None, uses the project root.
    """
    fig = plot_contours(cfg, material, metrics)
    
    if output_dir is None:
        output_dir = cfg['dir']['project_root']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    metrics_str = '_'.join(metrics)
    fig.savefig(os.path.join(output_dir, f"{material}_{metrics_str}_contours.png"), dpi=300, bbox_inches='tight')
    
    # Also save as PDF for publication quality
    fig.savefig(os.path.join(output_dir, f"{material}_{metrics_str}_contours.pdf"), bbox_inches='tight')
    
    plt.close(fig)