from typing import Dict, Tuple

from xasproc.io import read_xdi_file, read_athena_group
from xasproc.xas_spectrum import XASSpectrum
from xasproc.xas_collection import XASCollection
from xasproc.preprocess import energy_cut, get_reference, get_edge, shift_energy, normalize, calibrate_energy
import os

def load_spectrum(cfg: Dict, material: str, foil: XASSpectrum = None, e_range: Tuple =None) -> XASSpectrum:
    """
    Load and process a spectrum based on configuration.
    
    Args:
        spectrum_name (str, optional): Name of the spectrum to load. If None, uses the default from config.
        
    Returns:
        XASSpectrum: The normalized spectrum
    """

    source_file = cfg.get('data', {}).get('source_file')
    if source_file.endswith('.xdi'):
        X = load_spectrum_xdi(cfg)
    elif source_file.endswith('.prj'):
        X = load_spectrum_prj(cfg)

    # Create XASCollection
    XC = XASCollection([X], njobs=0)
    
    align_e0 = (foil.e0 if foil else None) or cfg.get('energy', {}).get('edge')
    e0 = cfg.get('energy', {}).get('edge')
    print(material, ': ', e0-200., e0+900)
    
    # Get normalization parameters from config
    norm_params = cfg.get('normalization')
    
    if foil is not None:
        # Get edge energy
        edge_energy = (
            XC
            .pipe(energy_cut, align_e0-200., align_e0+900)
            .pipe(get_reference)
            .pipe(get_edge)[0]
        )

        foil_edge = XASCollection([foil], njobs=0).pipe(get_edge)[0]
        energy_shifts = foil_edge - edge_energy

        N = (
            XC
            .pipe(shift_energy, energy_shifts)
            .pipe(calibrate_energy, foil, foil_edge)
            .pipe(energy_cut, e0-200., e0+800.)
            .pipe(normalize, norm_params, e0)
        )
    else:
        N = (
            XC
            .pipe(normalize, norm_params, e0)
        )

    # Get energy range from config
    energy = cfg.get('energy', {})
    if e_range is not None:
        e_min, e_max = e_range
    else:
        e_min = energy.get('min', e0-200.0)
        e_max = energy.get('max', e0+800.0)

    N = N.pipe(energy_cut, e_min, e_max)
    
    # Set edge energy
    N[0].e0 = e0
    
    return N[0]

def load_spectrum_prj(cfg: Dict) -> XASSpectrum:
    # Get data parameters from config
    data_cfg = cfg.get('data', {})
    # check energy column and mu expression are in the data cfg
    if 'energy_column' not in data_cfg or'mu_expression' not in data_cfg:
        raise ValueError("Energy column and mu expression must be in the data config")
    
    # Get file path
    file_path = data_cfg.get('source_file')
    group_name = data_cfg.get('athena_group')
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Spectrum file not found: {file_path}")
    
    # Load the XDI file
    xdi_info, D = read_athena_group(file_path, group_name)
    if 'eshift' in xdi_info:
        D['energy'] += xdi_info['eshift']
    
    # Create XASSpectrum object
    X = XASSpectrum.from_xdi(
        xdi_info=xdi_info, 
        xdi_data=D, 
        energy_column=data_cfg['energy_column'], 
        mu_expression=data_cfg['mu_expression'], 
        mu_ref_expression=data_cfg.get('mu_ref_expression', data_cfg['mu_expression'])
    )

    return X

def load_spectrum_xdi(cfg: Dict) -> XASSpectrum:
    # Get data parameters from config
    data_cfg = cfg.get('data', {})
    # check energy column and mu expression are in the data cfg
    if 'energy_column' not in data_cfg or'mu_expression' not in data_cfg:
        raise ValueError("Energy column and mu expression must be in the data config")
    
    # Get file path
    file_path = data_cfg.get('source_file')
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Spectrum file not found: {file_path}")
    
    # Load the XDI file
    xdi_info, D = read_xdi_file(file_path)
    
    # Create XASSpectrum object
    X = XASSpectrum.from_xdi(
        xdi_info=xdi_info, 
        xdi_data=D, 
        energy_column=data_cfg['energy_column'], 
        mu_expression=data_cfg['mu_expression'], 
        mu_ref_expression=data_cfg.get('mu_ref_expression', None)
    )

    return X
    