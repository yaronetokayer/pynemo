import numpy as np
from astropy.constants import G
from astropy import units as u
from tqdm import tqdm

def radial_mass_profile(r, positions, masses):
    """
    Computes the radial integrated mass profile given 3D Cartesian positions of particles and their masses.
    Assumes approximate spherical symmetry centered at the origin.

    Parameters:
    r (array-like): Radii at which to calculate the enclosed mass of the particles.
    positions (numpy.ndarray): 3D Cartesian positions of the particles (shape: (n, 3)).
    masses (numpy.ndarray): Masses of the particles (shape: (n,)).

    Returns:
    numpy.ndarray: Enclosed mass at each radius in `r`.
    """

    r_squared = np.square(r)
    distances_squared = np.sum(np.square(positions), axis=1)

    # Sort the particles by their distance from the origin
    sorted_indices = np.argsort(distances_squared)
    sorted_distances_squared = distances_squared[sorted_indices]
    sorted_masses = masses[sorted_indices]

    # Compute the cumulative sum of the sorted masses
    cumulative_masses = np.cumsum(sorted_masses)

    # Find the indices where each radius would fit into the sorted distances
    indices = np.searchsorted(sorted_distances_squared, r_squared, side='right') - 1

    # Use the indices to get the cumulative mass values
    integrated_mass = np.where(indices >= 0, cumulative_masses[indices], 0)

    return integrated_mass

def radial_density_profile(r, positions, masses, method='finite differences'):
    """
    Computes the radial 3D mass density profile given 3D Cartesian positions of particles and their masses.
    Assumes approximate spherical symmetry centered at the origin.

    Parameters:
    ----------
    r : array-like
        Radii at which to calculate the density of the particles.
    positions : numpy.ndarray
        3D Cartesian positions of the particles (shape: (n, 3)).
    masses : numpy.ndarray
        Masses of the particles (shape: (n,)).
    method : str, optional
        Method to compute the density profile. Options are 'finite differences' (default) and 'direct'.

    Returns:
    -------
    density : numpy.ndarray
        3D mass density at each radius in `r`.
    r_midpoints : numpy.ndarray, optional
        Radii corresponding to the density values when using the 'direct' method. Not returned for 'finite differences' method.

    Raises:
    ------
    ValueError
        If `r` contains zero when using the 'finite differences' method.
    """
    
    if method == 'direct':
        integrated_mass = radial_mass_profile(r, positions, masses)
        
        # Calculate the volume of spherical shells
        shell_volumes = (4/3) * np.pi * (r[1:]**3 - r[:-1]**3)
        
        # Compute the differential mass
        dM = np.diff(integrated_mass)
        
        # Compute density
        density = dM / shell_volumes
        
        # Midpoints for central differences
        r_midpoints = 0.5 * (r[1:] + r[:-1])
    
        return density, r_midpoints

    elif method == 'finite differences':
        # Ensure r does not contain zero to avoid division by zero
        if np.any(r == 0):
            raise ValueError("Radius array 'r' should not contain zero.")
        
        # Calculate the integrated mass profile
        integrated_mass = radial_mass_profile(r, positions, masses)
        
        # Compute the derivative of the integrated mass profile with respect to radius
        dM_dr = np.gradient(integrated_mass, r)
        
        # Compute the mass density by normalizing with the volume of the spherical shell
        density = dM_dr / (4 * np.pi * r**2)
        
        return density

def crossing_time_scale(r, positions, masses):
    """
    Computes the crossing time scale at a given radius for a halo,
    given 3D Cartesian positions of particles and their masses.
    Assumes approximate spherical symmetry centered at the origin.
    See e.g., Binney and Tremaine Eq. (2.40)

    Parameters:
    ----------
    r : array-like or float
        Radii (in kpc) at which to calculate the crossing time.
    positions : numpy.ndarray
        3D Cartesian positions of the particles in kpc (shape: (n, 3)).
    masses : numpy.ndarray
        Masses of the particles in Msun (shape: (n,)).

    Returns:
    -------
    tau : numpy.ndarray or float
        Crossing time scale in seconds corresponding to input `r` values.
    """
    
    # Ensure r is a numpy array if it's a list, otherwise keep it as is
    if isinstance(r, list):
        r = np.array(r)
    
    # Calculate enclosed mass at each radius
    m_enclosed = radial_mass_profile(r, positions, masses) * u.Msun
    
    # Calculate volume of spheres with radii r
    volume = (4 * np.pi * r**3 / 3) * u.kpc**3
    
    # Calculate average density within each radius
    rho_bar = m_enclosed / volume
    
    # Calculate crossing time scale
    tau = 1 / np.sqrt(G * rho_bar)
    
    # Return as float if single value, otherwise as numpy array
    if isinstance(tau, u.Quantity):
        tau_value = tau.to(u.s).value
        return tau_value if tau_value.size > 1 else float(tau_value)
    else:
        return float(tau.to(u.s).value)

def compute_length_unit_mc(positions, masses, n_iterations=1000, n_mask=100):
    r"""
    Computes the n body length unit \(\frac{1}{M^2}\sum_{i,j \neq i}^n \frac{m_i m_j}{|r_i - r_j|}\)
    using a Monte Carlo method. This approach estimates the term by randomly sampling subsets of particles
    to approximate the sum over all pairs.

    Parameters:
    ----------
    positions : numpy.ndarray
        3D Cartesian positions of the particles (shape: (n, 3)).
    masses : numpy.ndarray
        Masses of the particles (shape: (n,)).
    n_iterations : int, optional
        Number of Monte Carlo iterations to perform. Default is 1000.
    n_mask : int, optional
        Number of particles to sample in each iteration. Default is 100.

    Returns:
    -------
    float
        The estimated gravitational term.

    Notes:
    -----
    This method provides an approximation by sampling subsets of particles, which can be significantly
    faster than computing the term directly for large datasets. The accuracy of the result depends on
    the number of iterations and the size of the sampled subset.
    """

    # Number of particles
    n = len(masses)

    sum_term = np.zeros(n_iterations)
    for it in tqdm(range(n_iterations), desc="Monte Carlo Iterations"):
        
        # Randomly select a subset of particles
        mask = np.zeros(n, dtype=bool)
        mask[np.random.choice(n, n_mask, replace=False)] = True
        
        positions_masked = positions[mask]
        masses_masked = masses[mask]
        total_mass = np.sum(masses_masked)
        
        # Calculate pairwise distances and sum the terms
        for i in range(n_mask):
            for j in range(n_mask):
                if i != j:
                    distance = np.linalg.norm(positions_masked[i] - positions_masked[j])
                    sum_term[it] += masses_masked[i] * masses_masked[j] / distance
    
        # Normalize by the squared total mass
        sum_term[it] = total_mass**2 / sum_term[it]

    return np.mean(sum_term)