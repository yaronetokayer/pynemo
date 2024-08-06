import os
import numpy as np
import time
import threading
from textwrap import dedent
import agama

def get_dynamical_data(snap, n):
    '''
    Get arrays of time, position, velocity, and mass from a NEMO snapshot

    Inputs:
    snap: snapshot object from open_nemo
    n: the snapshot number from the snap file

    Returns:
    t, x, v, m
    '''
    t = snap[f'SnapShot/{n}/Parameters/Time'].data
    x = snap[f'SnapShot/{n}/Particles/Position'].data
    v = snap[f'SnapShot/{n}/Particles/Velocity'].data
    m = snap[f'SnapShot/{n}/Particles/Mass'].data

    return t, x, v, m

### From agama tutorial

def scalarize(x):
    return x if len(x)>1 else x[0]

def v_escape(pot, r):
    return scalarize((-pot.potential(np.column_stack((r, r*0, r*0))) * 2) ** 0.5)

def v_circ(pot, r):
    return scalarize((-r * pot.force(np.column_stack((r, r*0, r*0)))[:,0]) ** 0.5)

### Modified from Zhaozhou Li

def sample(n, den, pot, beta=0, r_a=np.inf, symmetrize=True):
    """
    Sample particles from a given density profile
    Modified from 

    Arguments:
    n: Number of particles to sample (int)
    den: Agama Density object
    pot: Agama Potential object
    beta: Anisotropy parameter for the distribution function. Default is 0.
    r_a: Anisotropy radius. Default is infinity.
    symmetrize: Boolean indicating whether to symmetrize the sample (i.e., make sure that for each particle there is another particle at the mirrored position). Default is True.

    Returns:
    xv: Sampled positions and velocities
    m: Particle masses
    """
    af = agama.ActionFinder(pot)
    df = agama.DistributionFunction(type='quasispherical', density=den, potential=pot, beta0=beta, r_a=r_a)
    mod = agama.GalaxyModel(potential=pot, df=df, af=af)

    if symmetrize:
        n1 = n // 2
        n2 = n - n1  # n2 >= n1
        xv, m = mod.sample(n2)
        m *= float(n2) / n
        xv = np.vstack([xv[:n1], -xv[:n2]])
        m = np.hstack([m[:n1], m[:n2]])
    else:
        xv, m = mod.sample(n)

    return xv, m

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


def timer_func(start_time, stop_event):
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        days = int(elapsed_time // 86400)
        hours = int((elapsed_time % 86400) // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        print(f"\rElapsed time: {days}d {hours}h {minutes}m {seconds:.2f}s", end="")
        time.sleep(1)

def run_gyrfalcon(**kwargs):
    '''
    This function sets up and submits a job to run the gyrfalcON N-body simulation code.

    Arguments:
    kwargs: Dictionary of keyword arguments for the gyrfalcON command. This includes parameters such as input file, output file, and other simulation settings.
    '''
    start_time = time.time()
    stop_event = threading.Event()
    threading.Thread(target=timer_func, args=(start_time, stop_event), daemon=True).start()

    if 'in' not in kwargs:
        kwargs['in'] = kwargs.pop('in_', '-')
    if os.path.exists(kwargs['out']):
        os.remove(kwargs['out'])
        print(f"The file {kwargs['out']} has been deleted to avoid overwrite error.\n")
    else:
        print(f"The file {kwargs['out']} does not yet exist. Proceeding with command.\n")
    val = " ".join([f"{k}={v}" for k, v in kwargs.items()])
    cmd = f"gyrfalcON {val}"

    script_template = dedent(r"""
    #!/bin/zsh
    
    source /Users/yaronetokayer/nemo/nemo_start.sh

    {cmd}
    """)

    script = script_template.format(cmd=cmd)
    print('running:\n', script)
    os.system(script)

    stop_event.set()  # Signal the timer to stop
    elapsed_time = time.time() - start_time
    days = int(elapsed_time // 86400)
    hours = int((elapsed_time % 86400) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"\nTotal elapsed time: {days}d {hours}h {minutes}m {seconds:.2f}s")
