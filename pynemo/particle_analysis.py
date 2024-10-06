import numpy as np
from astropy.constants import G
from astropy import units as u
from scipy.integrate import quad
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

def spherical_velocity_data(positions_velocities):
    """
    Computes the radial and tangential components of velocity,
    the velocity dispersion, and the anisotropy parameter beta
    for a system of particles.
    
    Args:
    - positions_velocities (numpy.ndarray): Array of particles' positions and velocities
      Shape: (N, 6), where first 3 columns are x, y, z positions, and last 3 columns are vx, vy, vz velocities.
    
    Returns:
    - v (numpy.ndarray): Speed of each particle.
    - v_r (numpy.ndarray): Radial component of velocity for each particle.
    - v_t (numpy.ndarray): Tangential component of velocity for each particle.
    - sigma_v_r (float): Velocity dispersion of the radial velocities.
    - sigma_v_t (float): Velocity dispersion of the tangential velocities.
    - beta (float): Anisotropy parameter.
    """

    # Split positions and velocities from the input array
    positions = positions_velocities[:, :3]
    velocities = positions_velocities[:, 3:]

    # Calculate radial distances
    radial_distances = np.linalg.norm(positions, axis=1)
    
    # Calculate dot product of position and velocity vectors
    dot_product = np.einsum('ij,ij->i', positions, velocities)
    
    # Radial velocity component
    v_r = dot_product / radial_distances
    
    # Total velocity magnitude
    v = np.linalg.norm(velocities, axis=1)
    
    # Tangential velocity component
    v_t = np.sqrt(v**2 - v_r**2)
    
    # Velocity dispersion (rms)
    sigma_v_r2 = np.mean(v_r**2)
    sigma_v_t2 = np.mean(v_t**2)
    
    # Anisotropy parameter beta
    beta = 1 - (sigma_v_t2 / (2 * sigma_v_r2))
    
    return {
        'speeds': v,
        'v_r': v_r, 
        'v_t': v_t,
        'sigma_r': np.sqrt(sigma_v_r2),
        'sigma_t': np.sqrt(sigma_v_t2),
        'beta': beta
    }

def beta(positions_velocities):
    r"""
    Computes the anisotropy parameter beta for a system of particles.
    \beta = 1 - \frac{<v_t^2>}{2<v_r^2>}
    
    Args:
    - positions_velocities (numpy.ndarray): Array of particle positions and velocities
      Shape: (N, 6), where first 3 columns are x, y, z positions, and last 3 columns are vx, vy, vz velocities.
    
    Returns:
    - beta (float): Anisotropy parameter.
    """

    # Split positions and velocities from the input array
    positions = positions_velocities[:, :3]
    velocities = positions_velocities[:, 3:]

    # Calculate radial distances
    radial_distances = np.linalg.norm(positions, axis=1)
    
    # Calculate dot product of position and velocity vectors
    dot_product = np.einsum('ij,ij->i', positions, velocities)
    
    # Radial velocity component
    v_r = dot_product / radial_distances
    
    # Total velocity magnitude
    v = np.linalg.norm(velocities, axis=1)
    
    # Magnitude of tangential velocity component
    v_t = np.sqrt(v**2 - v_r**2)
    
    # Velocity rms
    v_r_rms = np.mean(v_r**2)
    v_t_rms = np.mean(v_t**2)
    
    # Anisotropy parameter beta
    beta = 1 - (v_t_rms / (2 * v_r_rms))
    
    return beta

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

def mass_energy_distribution(positions_velocities, masses, potential, bins=100):
    r"""
    Computes the distribution function dM/dE for a system of particles.
    E is the binding energy per unit mass, defined as \Phi(r) - v^2/2,
    for a particle at 3D position r.

    Parameters:
    ----------
    positions_velocities : numpy.ndarray
        Array containing 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
        The first three columns are positions, and the last three columns are velocities.
        Assumes consistent units for position and velocity.
    masses : numpy.ndarray
        Masses of the particles (shape: (n,)).
    potential : agama.Potential
        Agama potential object used to compute the potential at given 3D locations.
    bins : int, optional
        Number of bins for the histogram of the binding energy. Default is 100.

    Returns:
    -------
    dNdE : numpy.ndarray
        Histogram values in units of mass per unit energy.
    bin_midpoints : numpy.ndarray
        Bin midpoints for the histogram of the binding energy.
        In units of velocity squared (specific energy).
    """

    # Split positions and velocities
    positions = positions_velocities[:, :3]
    velocities = positions_velocities[:, 3:]

    # Compute the potential at each position
    potential_values = potential.potential(positions)

    # Compute the kinetic energy per unit mass for each particle
    kinetic_energy = 0.5 * np.sum(velocities**2, axis=1)

    # Compute the binding energy per unit mass for each particle
    binding_energy = potential_values + kinetic_energy

    # Create the histogram of binding energies
    hist, bin_edges = np.histogram(binding_energy, bins=bins, weights=masses, density=False)

    bin_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    dE = np.diff(bin_midpoints)[0]

    return hist / dE, bin_midpoints

def density_of_states(E_array, potential, r_trunc=1000, epsrel=1e-8):
    """
    Computes the density of states g(E) for a spherically symmetric system potential.

    g(E) is defined as:
    g(E) = (4 * pi)^2 * integral from 0 to r_max(E) of r^2 * sqrt(2 * (E - Phi(r))) dr,
    where r_max(E) is the radius where E = Phi(r).  See, e.g., 4.56 in Binney and Tremaine

    Assumes a spherically symmetric potential, where Phi(r) = Phi(x, y, z) with x=r, y=0, z=0.

    Parameters:
    ----------
    E_array : numpy.ndarray
        Array of energies for which to compute the density of states (shape: (m,)).
    potential : agama.Potential
        Agama potential object used to compute the potential at given 3D locations.
    r_trunc: float, optional
        Radius at which to truncate the r_max for evaluating the integral.  See Natarajan, Hjorth, and Van Kampen (1997). Default is 1000
    epsrel : float, optional
        Relative error tolerance for the numerical integration. Default is 1e-8.

    Returns:
    -------
    gE : numpy.ndarray
        Density of states g(E) corresponding to input energy values.
    """
    # Define the integrand for g(E)
    def integrand(r, E):
        # Assume spherical symmetry: compute potential at (r, 0, 0)
        phi_r = potential.potential(np.array([[r, 0, 0]]))
        return np.sqrt(2 * np.abs(E - phi_r)) * r**2

    # Vectorize the computation of g(E) over the energy array
    gE = np.array([
        16 * np.pi**2 * quad(integrand, 0, _find_r_max(E, potential, r_trunc), args=(E,), epsrel=epsrel, limit=1000)[0]
        for E in E_array
    ])

    return gE

def _find_r_max(E, potential, r_high=100, r_trunc=10000, tol=1e-8, max_iter=100):
    """
    Finds the maximum radius r_max(E) where E = Phi(r) using the bisection method,
    or returns r_trunc if no valid solution is found. See Natarajan, Hjorth, and Van Kampen (1997).

    Parameters:
    ----------
    E : float
        The energy value for which to find the maximum radius.
    potential : agama.Potential
        The potential object used to compute the potential at a given 3D location.
    r_high: float, optional
        Initial guess for the computation of r_max using the bisection method.  Default is 100.
    r_trunc : float, optional
        The maximum possible value for r_max. Default is 10000.
    tol : float, optional
        Tolerance for convergence. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations for the bisection method. Default is 100.

    Returns:
    -------
    r_max : float
        The radius where E = Phi(r) or r_trunc if no valid range for bisection is found.

    Raises:
    ------
    RuntimeError:
        If the method does not converge within the maximum number of iterations.
    """
    # Define the function to find the root
    def to_solve(r):
        # Reshape r to a 2D array for the potential function
        r = np.array([[r, 0, 0]])
        return potential.potential(r)[0] - E

    # Set initial bounds for the bisection method
    r_low, r_high = 0, min(r_high, r_trunc)  # Start with an initial guess range

    # Ensure that we have a valid range for bisection (signs must be opposite)
    while to_solve(r_low) * to_solve(r_high) > 0:
        r_high *= 2  # Expand the upper bound until we find a valid range
        if r_high > r_trunc:
            # If we exceed r_trunc and still don't have a valid range, return r_trunc
            return r_trunc

    # Bisection method
    for _ in range(max_iter):
        r_mid = (r_low + r_high) / 2.0
        if to_solve(r_mid) > 0:
            r_high = r_mid
        else:
            r_low = r_mid

        # Check for convergence
        if r_high - r_low < tol:
            return min(r_mid, r_trunc)

    # If the method did not converge, raise an error
    raise RuntimeError(f"Bisection method did not converge for E={E} within {max_iter} iterations.")

def distribution_function_e(positions_velocities, masses, potential, bins=100, r_trunc=10000, epsrel=1e-8):
    """
    Isotropic distribution function f(E) for particle data, given an agama potential

    Parameters:
    ----------
    positions_velocities : numpy.ndarray
        Array containing 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
        The first three columns are positions, and the last three columns are velocities.
        Assumes consistent units for position and velocity.
    masses : numpy.ndarray
        Masses of the particles (shape: (n,)).
    potential : agama.Potential
        Agama potential object used to compute the potential at given 3D locations.
    bins : int, optional
        Number of bins for the histogram of the binding energy. Default is 100.
    r_trunc: float, optional
        Radius at which to truncate the r_max for evaluating the integral.  See Natarajan, Hjorth, and Van Kampen (1997). Default is 10000
    epsrel : float, optional
        Relative error tolerance for the numerical integration. Default is 1e-8.


    Returns:
    -------
    f_e : numpy.ndarray
        Histogram values of f(E).
    bin_midpoints : numpy.ndarray
        Bin midpoints for the histogram of the binding energy.
        In units of velocity squared.
    """

    dMdE, bin_midpoints = mass_energy_distribution(positions_velocities, masses, potential, bins=bins)
    g_e = density_of_states(bin_midpoints, potential, r_trunc=r_trunc, epsrel=epsrel)
    f_e = dMdE / g_e

    return f_e, bin_midpoints

def compute_actions(positions_velocities, potential):
    """
    Computes the radial action (J_r) and azimuthal action (L_z)
    for a set of particles in a spherical system using Agama potential.

    Args:
    - positions_velocities (numpy.ndarray): Array of particles' positions and velocities.
      Shape: (N, 6), where first 3 columns are x, y, z positions and last 3 columns are vx, vy, vz velocities.
    - potential: Agama potential object with a method that computes potential at given 3D cartesian coordinates.

    Returns:
    - J_r (numpy.ndarray): Radial action of each particle.
    - L_z (numpy.ndarray): Azimuthal action (z-component of angular momentum) of each particle.
    """

    # Split positions and velocities from the input array
    positions = positions_velocities[:, :3]
    velocities = positions_velocities[:, 3:]

    # Extract individual coordinates
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]

    # Compute the z-component of angular momentum
    L_z = x * vy - y * vx

    # Compute radius and speeds
    r = np.sqrt(x**2 + y**2 + z**2)
    v_squared = vx**2 + vy**2 + vz**2

    # Compute potential energies using the Agama potential object
    potential_values = potential.potential(positions)
    
    # Compute total energies (E = T + Phi)
    E = 0.5 * v_squared + potential_values
    
    def integrand(r, E, L, potential):
        return np.sqrt(2 * (E - potential.potential([r, 0, 0])) - (L**2 / r**2))

    J_r = np.zeros(len(E))
    for i in range(len(E)):  # Iterate over each particle to compute J_r
        if E[i] < 0:  # bound orbit condition (total energy less than zero)
            # Find periapsis and apoapsis for the given E and L
            # Here, we approximate, but typically requires solving for precise turning points
            r_peri = 1 / (2 * E[i] + 2 * np.abs(potential_values[i]))  # Approx peri but real impl requires nr solver
            r_apo = 1 / (2 * E[i] - 2 * np.abs(potential_values[i]))  # Approx apoapsis
            
            J_r[i] = (2 / np.pi) * quad(integrand, r_peri, r_apo, args=(E[i], L_z[i], potential), limit=100)[0]

    return J_r, L_z
