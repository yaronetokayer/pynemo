import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import G
from astropy import units as u
from scipy.integrate import quad
from tqdm import tqdm
import warnings

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import numpy as np

def radial_mass_profile(positions_velocities, masses, num_particles_per_bin=2500, return_bin_types=False):
    """
    Computes the radial integrated mass profile given 3D Cartesian positions of particles and their masses.
    Optionally returns a list of boolean values indicating whether bins are adaptive or fixed.

    Parameters:
    ----------
    positions_velocities (numpy.ndarray): 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
    masses (numpy.ndarray): Masses of the particles (shape: (n,)).
    num_particles_per_bin (int, optional): Number of particles per bin. Default is 2500.
    return_bin_types (bool, optional): Whether to return a list indicating adaptive/fixed bins. Default is False.

    Returns:
    -------
    tuple of numpy.ndarray: Radii at the end of each bin and enclosed mass at each radius.
    (optional) list of bool: List indicating adaptive (True) or fixed (False) bins.
    """
    positions = positions_velocities[:, :3]
    center_of_mass = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    positions -= center_of_mass

    distances_squared = np.sum(np.square(positions), axis=1)
    sorted_indices = np.argsort(distances_squared)
    sorted_distances = np.sqrt(distances_squared[sorted_indices])
    sorted_masses = masses[sorted_indices]

    cumulative_masses = np.cumsum(sorted_masses)
    inner_bin_sizes = [num_particles_per_bin // factor for factor in [100, 50, 25, 5, 2, 1]]

    num_particles = len(masses)
    bin_edges = []
    integrated_mass = []
    bin_types = []

    for bin_size in inner_bin_sizes:
        if bin_size < num_particles:
            bin_edges.append(sorted_distances[bin_size - 1])
            integrated_mass.append(cumulative_masses[bin_size - 1])
            bin_types.append(True)  # Adaptive bin
        else:
            break

    last_index = inner_bin_sizes[-1] if inner_bin_sizes[-1] < num_particles else num_particles_per_bin
    while last_index < num_particles:
        next_index = last_index + num_particles_per_bin
        if next_index < num_particles:
            bin_edges.append(sorted_distances[next_index - 1])
            integrated_mass.append(cumulative_masses[next_index - 1])
            bin_types.append(False)  # Fixed bin
        else:
            bin_edges.append(sorted_distances[-1])
            integrated_mass.append(cumulative_masses[-1])
            bin_types.append(False)  # Fixed bin
            break
        last_index = next_index

    if return_bin_types:
        return np.array(bin_edges), np.array(integrated_mass), bin_types
    return np.array(bin_edges), np.array(integrated_mass)

def radial_density_profile(positions_velocities, masses, num_particles_per_bin=2500, num_outer_subbins=3, return_mass_profile=False):
    """
    Computes the density profile for a spherical system of particles.
    Uses average density for inner bins and finite difference for fixed bins.
    The outermost bin is subdivided into logarithmically spaced subbins. 

    Parameters:
    ----------
    positions_velocities (numpy.ndarray): 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
    masses (numpy.ndarray): Masses of the particles (shape: (n,)).
    num_particles_per_bin (int, optional): Number of particles per bin. Default is 2500.
    num_outer_subbins (int, optional): Number of logarithmically spaced subbins in the outermost bin. Default is 3.

    Returns:
    -------
    tuple of numpy.ndarray: Radii of the bins and corresponding densities.
    """
    radii, enclosed_mass, bin_types = radial_mass_profile(positions_velocities, masses, num_particles_per_bin, return_bin_types=True)

    inner_radii = []
    inner_densities = []
    fixed_radii = []
    fixed_densities = []

    # Compute particle distances for use later
    positions = positions_velocities[:, :3]
    distances = np.sqrt(np.sum(np.square(positions), axis=1))

    for i, is_adaptive in enumerate(bin_types):
        if is_adaptive:
            volume = (4 / 3) * np.pi * radii[i]**3
            density = enclosed_mass[i] / volume
            inner_radii.append(radii[i])
            inner_densities.append(density)
        else:
            if i > 0:
                delta_m = enclosed_mass[i] - enclosed_mass[i - 1]
                shell_volume = (4 / 3) * np.pi * (radii[i]**3 - radii[i - 1]**3)
                density = delta_m / shell_volume

                # Calculate the median distance for the particles in the current bin
                bin_indices = (distances > radii[i - 1]) & (distances <= radii[i])
                median_distance = np.median(distances[bin_indices])

                fixed_radii.append(median_distance)
                fixed_densities.append(density)

    # Handle the outermost bin with subbins to improve the resolution of the density profile
    if len(fixed_radii) > 0:
        r_outer_min, r_outer_max = fixed_radii[-2], fixed_radii[-1]
        subbin_edges = np.logspace(np.log10(r_outer_min), np.log10(r_outer_max), num_outer_subbins + 1)
        subbin_radii = []
        subbin_densities = []

        for j in range(num_outer_subbins):
            subbin_mask = (distances > subbin_edges[j]) & (distances <= subbin_edges[j + 1])
            subbin_distances = distances[subbin_mask]
            subbin_mass = np.sum(masses[subbin_mask])
            subbin_volume = (4 / 3) * np.pi * (subbin_edges[j + 1]**3 - subbin_edges[j]**3)
            
            if len(subbin_distances) > 0:
                subbin_radii.append(np.median(subbin_distances))
                subbin_densities.append(subbin_mass / subbin_volume)

        fixed_radii = np.concatenate((fixed_radii[:-1], subbin_radii))
        fixed_densities = np.concatenate((fixed_densities[:-1], subbin_densities))

    all_radii = np.concatenate((inner_radii, fixed_radii))
    all_densities = np.concatenate((inner_densities, fixed_densities))

    if return_mass_profile:
        return all_radii, all_densities, radii, enclosed_mass
    else:
        return all_radii, all_densities


def integrated_mass(r, positions_velocities, masses):
    """
    Computes the radial integrated mass profile given 3D Cartesian positions of particles and their masses.
    Assumes approximate spherical symmetry centered at the origin.

    Parameters:
    r (array-like): Radii at which to calculate the enclosed mass of the particles.
    positions_velocities (numpy.ndarray): 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
    masses (numpy.ndarray): Masses of the particles (shape: (n,)).

    Returns:
    numpy.ndarray: Enclosed mass at each radius in `r`.
    """

    positions = positions_velocities[:, :3]
    center_of_mass = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    positions -= center_of_mass

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

def average_density(r, positions_velocities, masses):
    """
    Computes the radial average density profile given 3D Cartesian positions of particles and their masses.
    Assumes approximate spherical symmetry centered at the origin.

    Parameters:
    r (array-like): Radii at which to calculate the enclosed mass of the particles.
    positions_velocities (numpy.ndarray): 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
    masses (numpy.ndarray): Masses of the particles (shape: (n,)).

    Returns:
    numpy.ndarray: Enclosed mass at each radius in `r`.
    """

    average_density = 3 * integrated_mass(r, positions_velocities, masses) / 4 / np.pi / r**3

    return average_density

# def radial_density_profile(r, positions_velocities, masses, method='finite differences'):
#     """
#     Computes the radial 3D mass density profile given 3D Cartesian positions of particles and their masses.
#     Assumes approximate spherical symmetry centered at the origin.

#     Parameters:
#     ----------
#     r : array-like
#         Radii at which to calculate the density of the particles.
#     positions_velocities : numpy.ndarray
#         3D Cartesian positions and velocities of the particles (shape: (n, 3)).
#     masses : numpy.ndarray
#         Masses of the particles (shape: (n,)).
#     method : str, optional
#         Method to compute the density profile. Options are 'finite differences' (default) and 'direct'.

#     Returns:
#     -------
#     density : numpy.ndarray
#         3D mass density at each radius in `r`.
#     r_midpoints : numpy.ndarray, optional
#         Radii corresponding to the density values when using the 'direct' method. Not returned for 'finite differences' method.

#     Raises:
#     ------
#     ValueError
#         If `r` contains zero when using the 'finite differences' method.
#     """
    
#     if method == 'direct':
#         integrated_mass = radial_mass_profile(r, positions_velocities, masses)
        
#         # Calculate the volume of spherical shells
#         shell_volumes = (4/3) * np.pi * (r[1:]**3 - r[:-1]**3)
        
#         # Compute the differential mass
#         dM = np.diff(integrated_mass)
        
#         # Compute density
#         density = dM / shell_volumes
        
#         # Midpoints for central differences
#         r_midpoints = 0.5 * (r[1:] + r[:-1])
    
#         return density, r_midpoints

#     elif method == 'finite differences':
#         # Ensure r does not contain zero to avoid division by zero
#         if np.any(r == 0):
#             raise ValueError("Radius array 'r' should not contain zero.")
        
#         # Calculate the integrated mass profile
#         integrated_mass = radial_mass_profile(r, positions_velocities, masses)
        
#         # Compute the derivative of the integrated mass profile with respect to radius
#         dM_dr = np.gradient(integrated_mass, r)
        
#         # Compute the mass density by normalizing with the volume of the spherical shell
#         density = dM_dr / (4 * np.pi * r**2)
        
#         return density

def thin_xy_slice(positions_velocities, width_percentage=0.05):
    """
    Extract a thin slice in the x-y plane centered at z=0.

    Parameters:
    ----------
    positions_velocities (numpy.ndarray): 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
    width_percentage (float, optional): The width of the slice as a percentage of the full radius. Default is 5%.

    Returns:
    -------
    numpy.ndarray: Array of positions_velocities within the thin slice in the x-y plane.
    """
    
    positions = positions_velocities[:, :3]

    # Compute the full radius of the spherical system (maximum distance from origin)
    distances = np.linalg.norm(positions, axis=1)
    max_radius = np.max(distances)

    # Compute the width of the slice in the z-direction
    slice_half_width = width_percentage * max_radius / 2

    # Filter particles based on the z-coordinate
    mask = np.abs(positions[:, 2]) < slice_half_width

    # Return the subset of positions_velocities that fall within the slice
    return positions[mask][:, :2]

def r_200(positions_velocities, masses, rho_crit=None, cosmo=cosmo, z=0, g=G):
    """
    Compute the radius (r_200) within which the average density is 200 times the critical density
    using a bisection method.

    Parameters:
    ----------
    positions_velocities : numpy.ndarray
        Array containing 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
        The first three columns are positions, and the last three columns are velocities.
        Assumes position in kpc if rho_crit is None.
    masses : numpy.ndarray
        Array containing the masses of the particles (shape: (n,)).
        Assumes units of Msun if rho_crit is None.
    rho_crit : float, optional
        Critical density in custom units. If None, it will be computed using the cosmological model and redshift.
    cosmo : astropy.cosmology.FLRW, optional
        Cosmological model used to compute the critical density if rho_crit is None. Default is FlatLambdaCDM(H0=70, Om0=0.3).
    z : float, optional
        Redshift at which to compute the critical density if rho_crit is None. Default is 0 (present day).

    Returns:
    -------
    float
        The radius (r_200) within which the average density is 200 times the critical density.

    Raises:
    ------
    ValueError
        If no solution is found within the provided interval or if the bisection method does not converge
        within the maximum number of iterations.

    Notes:
    -----
    - The function uses a bisection method to iteratively find the radius r_200 where the average density
      within that radius is 200 times the critical density of the universe at the given redshift.
    - The critical density is computed using the provided cosmological model and redshift.
    - If the density difference function does not change sign over the interval defined by the minimum
      and maximum radii of the particles, a ValueError is raised indicating no solution within the interval.
    """

    if rho_crit == None:
        # Compute the critical density at redshift z
        rho_crit = (3 * cosmo.H(z)**2 / (8 * np.pi * g)).to(u.Msun / u.kpc**3).value

    # Assuming positions are in kpc and masses in Msun
    positions = positions_velocities[:, :3]

    # Define the function to find the average density
    def density_difference(r):
        enclosed_mass = integrated_mass(r, positions_velocities, masses)
        average_density = enclosed_mass / (4/3 * np.pi * r**3)
        return average_density - 200 * rho_crit

    # Use the next to minimum and 1000 times the maximum radii from the positions_velocities array
    r_guess_min = np.sort(np.linalg.norm(positions, axis=1))[2]
    r_guess_max = 1000 * np.max(np.linalg.norm(positions, axis=1))

    # Bisection method to find r_200
    tolerance = 1e-4  # Set the tolerance for convergence
    max_iterations = 100  # Set the maximum number of iterations

    def bisection_method(func, a, b, tol, max_iter):
        fa, fb = func(a), func(b)
        if fa * fb > 0:
            raise ValueError("No solution found within the provided interval.")
        
        for i in range(max_iter):
            c = (a + b) / 2
            fc = func(c)
            
            if abs(fc) < tol or abs(b - a) < tol:
                return c
            
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        
        raise ValueError("Bisection method did not converge within the maximum number of iterations.")
    
    r_200 = bisection_method(density_difference, r_guess_min, r_guess_max, tolerance, max_iterations)
    return r_200

def n_200(positions_velocities, masses, cosmo=cosmo, z=0):
    """
    Compute the number of particles within r_200 where the average density is 200 times the critical density.

    Parameters:
    positions_velocities (numpy.ndarray): Array of particles' positions and velocities.
    masses (numpy.ndarray): Masses of the particles.

    Returns:
    int: Number of particles within r_200.
    """

    # Compute r_200
    r200_radius = r_200(positions_velocities, masses, cosmo, z)

    # Compute the distances of all particles from the origin
    positions = positions_velocities[:, :3]
    distances = np.linalg.norm(positions, axis=1)

    # Count the number of particles within r_200
    n_particles_within_r200 = np.sum(distances < r200_radius)

    return int(n_particles_within_r200)

def epsilon_zhang(positions_velocities, masses, cosmo=cosmo, z=0, alpha=2):
    """
    Compute the optimal softening length (epsilon), assuming a Plummer kernel,
    using the prescription of Zhang et al. (2019).

    Parameters:
    ----------
    positions_velocities : numpy.ndarray
        Array containing 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
        The first three columns are positions, and the last three columns are velocities.
        Assumes consistent units for position and velocity.
    masses : numpy.ndarray
        Array containing the masses of the particles (shape: (n,)).
    cosmo : astropy.cosmology.FLRW, optional
        Cosmological model used to compute the critical density and <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>r</mi><mn>200</mn></msub></mrow><annotation encoding="application/x-tex">r_{200}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.5806em;vertical-align:-0.15em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right:0.02778em;">r</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height:0.3011em;"><span style="top:-2.55em;margin-left:-0.0278em;margin-right:0.05em;"><span class="pstrut" style="height:2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">200</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height:0.15em;"><span></span></span></span></span></span></span></span></span></span>. Default is FlatLambdaCDM(H0=70, Om0=0.3).
    z : float, optional
        Redshift at which to compute the critical density. Default is 0 (present day).
    alpha : float, optional
        Scaling factor for the softening length. Default is 2.

    Returns:
    -------
    float
        Optimal softening length (epsilon).

    References:
    ----------
    Zhang et al. (2019). "Optimal softening length for collisionless N-body simulations".
    Monthly Notices of the Royal Astronomical Society.
    https://doi.org/10.1093/mnras/stz1370
    """

    r200 = r_200(positions_velocities, masses, cosmo, z)
    n200 = n_200(positions_velocities, masses, cosmo, z)

    return alpha * r200 / np.sqrt(n200)

def estimate_mean_interparticle_separation(positions_velocities):
    """
    Estimate the mean interparticle separation in a spherical system of particles based on their positions.

    Parameters
    ----------
    positions_velocities : numpy.ndarray
        A (N, 6) array where the first three elements of each row represent the Cartesian positions (x, y, z) 
        of the particles, and the next three elements represent the velocity components. 
        Only the positions are used in this function.
    
    Returns
    -------
    mean_separation : float
        Estimated mean interparticle separation (in the same units as the positions, e.g., kpc).
    
    Notes
    -----
    This function assumes a spherical system with the inner radius at 0 and estimates the outer radius as 
    the maximum radial distance from the origin. The particles are assumed to be uniformly distributed within
    this spherical volume, and the mean separation is approximated by the cube root of the volume per particle.
    """
    
    # Extract the positions (first three columns of the array)
    positions = positions_velocities[:, :3]
    
    # Calculate the radial distance of each particle from the origin
    distances = np.linalg.norm(positions, axis=1)
    
    # Calculate the volume of the spherical system
    volume = (4/3) * np.pi * np.max(distances)**3
    
    # Number of particles
    num_particles = positions.shape[0]
    
    # Estimate the mean interparticle separation
    mean_separation = (volume / num_particles)**(1/3)
    
    return mean_separation

def v_esc(r, positions_velocities, masses, g=G):
    """
    Computes the escape velocity given 3D Cartesian positions of particles and their masses.
    Assumes approximate spherical symmetry centered at the origin.

    Parameters:
    r (array-like): Radii at which to calculate the escape velocity.
    positions_velocities (numpy.ndarray): 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
    masses (numpy.ndarray): Masses of the particles (shape: (n,)).
    g (float): value of gravitational constant in appropriate units

    Returns:
    numpy.ndarray: escape velocity as a function of r
    """
    m_tot = integrated_mass(r, positions_velocities, masses)

    return np.sqrt( 2 * g * m_tot / r)

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

def beta(positions_velocities, sampling_factor=1.0):
    r"""
    Computes the anisotropy parameter beta for a system of particles.
    \beta = 1 - \frac{<v_t^2>}{2<v_r^2>}
    
    Args:
    - positions_velocities (numpy.ndarray): Array of particle positions and velocities
      Shape: (N, 6), where first 3 columns are x, y, z positions, and last 3 columns are vx, vy, vz velocities.
    - sampling_factor (float, optional): Fraction of particles to sample. Default is 1.0 (use all particles).
    
    Returns:
    - beta (float): Anisotropy parameter.
    """
    if sampling_factor < 1.0:
        num_samples = int(len(positions_velocities) * sampling_factor)
        indices = np.random.choice(len(positions_velocities), num_samples, replace=False)
        positions_velocities = positions_velocities[indices]

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
    v_t_squared = v**2 - v_r**2
    # Check for significantly negative values in v_t_squared
    if np.any(v_t_squared < -1e-4):
        warnings.warn(
            "Detected negative tangential velocity squared values below -1e-3. "
            "This is unphysical. Clipping negative values to zero, but beta may not be accurate.",
            RuntimeWarning
        )
    v_t = np.sqrt(np.maximum(v_t_squared, 0))
    
    # Velocity rms
    v_r_rms = np.mean(v_r**2)
    v_t_rms = np.mean(v_t**2)

    # Anisotropy parameter beta
    beta = 1 - (v_t_rms / (2 * v_r_rms))
    
    return beta

def crossing_time_scale(r, positions_velocities, masses, g=G.to(u.kpc * u.km**2 / u.s**2 / u.Msun).value):
    """
    Computes the crossing time scale (in time units of input velocities) at a given radius for a halo,
    given 3D Cartesian positions of particles and their masses.
    Assumes approximate spherical symmetry centered at the origin.
    See e.g., Binney and Tremaine Eq. (2.40)

    Parameters:
    ----------
    r : array-like or float
        Radii at which to calculate the crossing time.
    positions_velocities : numpy.ndarray
        3D Cartesian positions and velocities of the particles (shape: (n, 6)).
    masses : numpy.ndarray
        Masses of the particles (shape: (n,)).
    g : float, optional
        Gravitational constant in appropriate units. Default is G in kpc, km/s, Msun.

    Returns:
    -------
    tau : numpy.ndarray or float
        Crossing time scale corresponding to input `r` values.
    """

    # Calculate average density within each radius
    rho_bar = average_density(r, positions_velocities, masses)
    
    # Calculate crossing time scale
    tau = 1 / np.sqrt(g * rho_bar)
    
    return tau if tau.size > 1 else float(tau)


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

    return f_e, dMdE, bin_midpoints

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


def subtract_com(positions_velocities, masses, chatter=True):
    """ 
    Subtract off center of mass positions and velocities.
    
    Parameters:
    ----------
    positions_velocities : numpy.ndarray
        Array containing 3D Cartesian positions and velocities of the particles (shape: (n, 6)).
        The first three columns are positions, and the last three columns are velocities.
    masses : numpy.ndarray
        1D array containing the masses of the particles.

    Returns:
    -------
    positions_velocities : numpy.ndarray
        Input positions_velocities with c.o.m. subtracted off.
    """

    positions = positions_velocities[:, :3].copy()
    com = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    positions -= com

    if chatter:
        print('Subtracted off c.o.m. position of ', com)
        print('New c.o.m. is ', np.sum(positions * masses[:, None], axis=0) / np.sum(masses))

    velocities = positions_velocities[:, 3:].copy()
    v_com = np.sum(velocities * masses[:, None], axis=0) / np.sum(masses)
    velocities -= v_com

    if chatter: 
        print('Subtracted off c.o.m. velocity of ', v_com)
        print('New c.o.m. velocity is ', np.sum(velocities * masses[:, None], axis=0) / np.sum(masses))

    return np.hstack((positions, velocities))
