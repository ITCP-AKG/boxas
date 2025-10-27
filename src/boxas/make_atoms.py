import pandas as pd

from random import random

from ase import Atoms
from ase.build import bulk, make_supercell
from ase.data import atomic_numbers
from ase.lattice.tetragonal import SimpleTetragonalFactory

import numpy as np

def create_cluster(element='Ni', structure='fcc', lc_a=3.52, lc_c=None, cluster_radius=8.0, 
                       supercell_size=(3, 3, 3)):
    """
    Create an FCC crystal cluster using ASE.
    
    Parameters:
    -----------
    element : str
        Chemical symbol of the element (default: 'Ni')
    structure : str
        Crystal structure (default: 'fcc')
    lc_a : float
        Lattice constant in Angstroms (default: 3.52 for Ni)
    lc_c : float
        Tetragonal lattice constant in Angstroms (default: 3.52 for Ni), if none lc_a is used
    cluster_radius : float
        Radius of the cluster in Angstroms (default: 8.0)
    supercell_size : tuple
        Size of supercell (nx, ny, nz) before cutting cluster (default: (3,3,3))
    center_atom_index : int or None
        Index of center atom. If None, uses geometric center
    
    Returns:
    --------
    cluster : ase.Atoms
        ASE Atoms object containing the cluster
    center_index : int
        Index of the central atom in the cluster
    """
    
    # Create FCC unit cell
    lc_c = lc_a if lc_c is None else lc_c
    unit_cell = bulk(element, structure, a=lc_a, b=lc_a, c=lc_c, cubic=True)
    
    # Create supercell
    nx, ny, nz = supercell_size
    supercell_matrix = [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]
    supercell = make_supercell(unit_cell, supercell_matrix)
    
    # Get positions and find center
    positions = supercell.get_positions()
    
    # Find atom closest to geometric center
    geometric_center = np.mean(positions, axis=0)
    distances_to_center = np.linalg.norm(positions - geometric_center, axis=1)
    center_atom_index = np.argmin(distances_to_center)
    
    center_position = positions[center_atom_index]
    
    # Calculate distances from center atom
    distances = np.linalg.norm(positions - center_position, axis=1)
    
    # Select atoms within cluster radius
    cluster_mask = distances <= cluster_radius
    cluster_indices = np.where(cluster_mask)[0]
    
    # Create cluster
    cluster_positions = positions[cluster_mask]
    cluster_symbols = [supercell.get_chemical_symbols()[i] for i in cluster_indices]
    
    cluster = Atoms(symbols=cluster_symbols, positions=cluster_positions)
    
    # Find new index of center atom in cluster
    center_index_in_cluster = np.where(cluster_indices == center_atom_index)[0][0]
    
    return cluster, center_index_in_cluster

class L1_0Factory(SimpleTetragonalFactory):
    "A factory for creating tetraenite (L1_0) lattices (tetragonal symmetry)."
    bravais_basis = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    element_basis = (0, 1, 1, 0)


# Now create a cluster by cutting a sphere
def create_tetragonal_cluster(
        elements=['Ni', 'Fe'], 
        lc_a=3.52, 
        c_factor=1., 
        cluster_radius=8.0, 
        supercell_size=(3, 3, 3)
    ):
    """
    Create a spherical cluster from a lattice structure.
    
    Parameters:
    -----------
    structure : ase.Atoms
        The lattice structure to cut from
    cluster_radius : float
        Radius of the cluster in Angstroms
    
    Returns:
    --------
    cluster : ase.Atoms
        The spherical cluster
    center_index : int
        Index of the central atom
    """
    L1_0 = L1_0Factory()

    # Create the lattice structure
    lattice_structure = L1_0(elements, latticeconstant={'a': lc_a, 'c': lc_a*c_factor})

    # Create a supercell to have enough atoms for cluster cutting
    supercell_matrix = [[8, 0, 0], [0, 8, 0], [0, 0, 8]]
    supercell = make_supercell(lattice_structure, supercell_matrix)

    positions = supercell.get_positions()
    
    # Find atom closest to geometric center
    geometric_center = np.mean(positions, axis=0)
    distances_to_center = np.linalg.norm(positions - geometric_center, axis=1)
    center_atom_index = np.argmin(distances_to_center)
    
    center_position = positions[center_atom_index]
    
    # Calculate distances from center atom
    distances = np.linalg.norm(positions - center_position, axis=1)
    
    # Select atoms within cluster radius
    cluster_mask = distances <= cluster_radius
    cluster_indices = np.where(cluster_mask)[0]
    
    # Create cluster
    cluster_positions = positions[cluster_mask]
    cluster_symbols = [supercell.get_chemical_symbols()[i] for i in cluster_indices]
    
    cluster = Atoms(symbols=cluster_symbols, positions=cluster_positions)
    
    # Find new index of center atom in cluster
    center_index_in_cluster = np.where(cluster_indices == center_atom_index)[0][0]
    
    return cluster, center_index_in_cluster


def cluster_to_xyz_string(cluster, center_index=0):
    """
    Convert ASE Atoms cluster to XYZ format string.
    
    Parameters:
    -----------
    cluster : ase.Atoms
        ASE Atoms object
    center_index : int
        Index of the central atom (for reference)
    
    Returns:
    --------
    xyz_string : str
        XYZ format string
    """
    positions = cluster.get_positions()
    symbols = cluster.get_chemical_symbols()
    
    # Center the cluster at the absorbing atom
    center_pos = positions[center_index]
    centered_positions = positions - center_pos
    
    xyz_lines = [str(len(cluster))]
    xyz_lines.append(f"FCC cluster centered at atom {center_index}")
    
    for i, (symbol, pos) in enumerate(zip(symbols, centered_positions)):
        xyz_lines.append(f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")
    
    return "\n".join(xyz_lines)

def cluster_to_feff_df(cluster, alloy=None, center_index=0):
    """
    Convert cluster to FEFF POTENTIALS and ATOMS cards format.
    
    Parameters:
    -----------
    cluster : ase.Atoms
        ASE Atoms object
    center_index : int
        Index of the central absorbing atom
    
    Returns:
    --------
    potentials_card : str
        FEFF POTENTIALS card
    atoms_card : str
        FEFF ATOMS card
    """
    positions = cluster.get_positions()
    symbols = cluster.get_chemical_symbols()
    
    # Center at absorbing atom
    center_pos = positions[center_index]
    centered_positions = positions - center_pos

    atoms_lines = []
    
    for i, (symbol, pos) in enumerate(zip(symbols, centered_positions)):
        distance = np.linalg.norm(pos)
        S = pd.Series({
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "ipot": 0 if i == center_index else 1,
            "symbol": symbol,
            "distance": distance
        })
        
        atoms_lines.append(S)
    
    A = pd.DataFrame(atoms_lines)

    if alloy is not None:
        AF = A.loc[A['ipot']!=0].sample(frac=alloy[1])
        AF[['ipot', 'symbol']] = [2, alloy[0]]
        A.loc[AF.index, :] = AF
    
    return A

def get_potentials_card(D):
    """
    Generate FEFF POTENTIALS card from cluster DataFrame.
    
    Parameters:
    -----------
    cluster_df : pandas.DataFrame
        DataFrame containing cluster information
    
    Returns:
    --------
    potentials_card : str
        FEFF POTENTIALS card
    """

    P = (
        D[['ipot', 'symbol']]
        .sort_values('ipot', ascending=True)
        .drop_duplicates()
    )

    P['Z'] = P['symbol'].apply(lambda x: atomic_numbers[x])
    P['l_scmt'] = 3
    P['l_fms'] = 3
    P['stoichiometry'] = P['ipot'].apply(lambda x: 1 if x != 0 else 0.0001)

    potentials_card = (
        P[['ipot', 'Z', 'symbol', 'l_scmt', 'l_fms','stoichiometry']]
        .to_csv(index=False, sep=' ', header=False)
    )

    potentials_card = "\nPOTENTIALS\n" + potentials_card + "\n"

    return potentials_card

def make_atoms_feff(alloy, fraction, c_factor=None):
    if c_factor is None:
        cluster, center_idx = create_cluster(
            element=alloy['elements'][0], 
            structure=alloy['structure'], 
            lc_a=alloy['lattice_const'], 
            cluster_radius=12.0,
            supercell_size=(8, 8, 8)
        )
    else:
        cluster, center_idx = create_tetragonal_cluster(
            elements=alloy['elements'], 
            lc_a=alloy['lattice_const'], 
            c_factor=c_factor, 
            cluster_radius=12.0,
            supercell_size=(8, 8, 8)
        )

    if c_factor is None:
        alloy_compound = None if len(alloy['elements']) == 1 else (alloy['elements'][1], fraction)
    else:
        alloy_compound = None

    D = cluster_to_feff_df(cluster, alloy=alloy_compound, center_index=center_idx)

    potentials = get_potentials_card(D)

    atoms = D.round(4).to_csv(index=False, header=False, sep=' ')

    return potentials + '\n\nATOMS\n' + atoms

def make_crystal_fdmnes(alloy, *args):
    structure = alloy['structure']
    lc_a = alloy['lattice_const']
    elements = alloy['elements']

    if len(elements) == 1:
        atoms = bulk(elements[0], structure, a=lc_a, cubic=True)
    else:
        # This part can be extended for binary/ternary alloys if needed
        raise NotImplementedError("FDMNES crystal generation for alloys is not implemented yet.")

    # Get lattice parameters
    cell_params = atoms.get_cell_lengths_and_angles()
    a, b, c, alpha, beta, gamma = cell_params

    # Start building the crystal description string
    crystal_str = "Crystal\n"
    crystal_str += f"    {a:.4f} {b:.4f} {c:.4f} {alpha:.2f} {beta:.2f} {gamma:.2f}\n"

    # Get atomic numbers and scaled positions (fractional coordinates)
    atomic_numbers = atoms.get_atomic_numbers()
    scaled_positions = atoms.get_scaled_positions()

    for z, pos in zip(atomic_numbers, scaled_positions):
        crystal_str += f" {z}  {pos[0]:.5f}   {pos[1]:.5f}   {pos[2]:.5f}\n"
    
    crystal_str += "\n! Convolution keyword : broadening with a width increasing versus energy as an arctangent\n"
    crystal_str += "\nConvolution\n"
    crystal_str += "\nEnd\n"

    return crystal_str
