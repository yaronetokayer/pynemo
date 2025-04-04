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
    # pot = snap[f'SnapShot/{n}/Particles/Potential'].data

    return t, x, v, m #, pot

def generate_spherical_shell(inner_radius, outer_radius, num_particles, masses, mean_v_r, v_disp):
    """
    Generates particle data (positions, velocities, and masses) for a spherical shell of particles.
    The particles are uniformly distributed in space within the spherical shell.
    Particles are given a net radial velocity, with an isotropic random velocity added.

    Parameters:
    ----------
    inner_radius : float
        Inner radius of the spherical shell.
    outer_radius : float
        Outer radius of the spherical shell.
    num_particles : int
        Number of particles to generate.
    masses : float or numpy.ndarray
        Masses of the particles in units of Msun. If a single number is provided, all particles will 
        have the same mass.
        If an array is provided, it should have length `num_particles`.
    mean_v_r : float
        Mean (inward) radial velocity (in km/s).
    v_disp : float
        Standard deviation of each of the three Gaussians from which the three velocity components are 
        drawn.

    Returns:
    -------
    numpy.ndarray
        (n, 6) array where the first three elements of each row are the Cartesian positions (in kpc)
        and the next three elements are the velocity components (in km/s).
    numpy.ndarray
        1D array of length `n` representing the masses of the particles (in Msun).
    """

    ### MASS ###

    if np.isscalar(masses):
        masses = np.full(num_particles, masses)
    elif len(masses) != num_particles:
        raise ValueError("Length of masses array must be 1, or match the number of particles.")

    ### POSITION ###

    # Generate random positions uniformly within the spherical shell
    radii = np.random.uniform(inner_radius**3, outer_radius**3, num_particles)**(1/3)
    phi = np.random.uniform(0, 2 * np.pi, num_particles)
    cos_theta = np.random.uniform(-1, 1, num_particles)
    theta = np.arccos(cos_theta)
    positions = np.vstack((radii * np.sin(theta) * np.cos(phi),
                           radii * np.sin(theta) * np.sin(phi),
                           radii * np.cos(theta))).T

    ### VELOCITY ###

    v_r = np.random.normal(-mean_v_r, v_disp, num_particles)
    v_theta = np.random.normal(0, v_disp, num_particles)
    v_phi = np.random.normal(0, v_disp, num_particles)

    # Convert spherical velocity components to Cartesian coordinates
    vx = v_r * np.sin(theta) * np.cos(phi) + v_theta * np.cos(theta) * np.cos(phi) - v_phi * np.sin(phi)
    vy = v_r * np.sin(theta) * np.sin(phi) + v_theta * np.cos(theta) * np.sin(phi) + v_phi * np.cos(phi)
    vz = v_r * np.cos(theta) - v_theta * np.sin(theta)
    velocities = np.vstack((vx, vy, vz)).T

    return np.hstack((positions, velocities)), masses  # Returns a (num_particles, 6) array with positions (x, y, z) and velocities (vx, vy, vz), and a 1D array of masses

def generate_spherical_shell_li(inner_radius, outer_radius, num_particles, masses, virial_velocity, **kwargs):
    r"""
    Generates particle data (positions, velocities, and masses) for a spherical shell of particles.
    The particles are uniformly distributed in space within the spherical shell
    Speeds are drawn from the Li+2020 log-normal distribution, and isotropic directions are assigned 
    randomly

    Parameters:
    ----------
    inner_radius : float
        Inner radius of the spherical shell (in kpc).
    outer_radius : float
        Outer radius of the spherical shell (in kpc).
    num_particles : int
        Number of particles to generate.
    masses : float or numpy.ndarray
        Masses of the particles in units of Msun. If a single number is provided, all particles will have the same mass.
        If an array is provided, it should have length `num_particles`.
    virial_velocity : float
        The virial velocity (in km/s) used in the 'Li+20' velocity profile.
    **kwargs :
        - mu1 : float, optional (default set by the infall_velocity_li function)
            Mean of the log-normal distribution used in the 'Li+20' velocity profile.
        - sig1 : float, optional (default set by the infall_velocity_li function)
            Standard deviation of the log-normal distribution used in the 'Li+20' velocity profile.
        
    Returns:
    -------
    numpy.ndarray
        A (num_particles, 6) array where the first three elements of each row are the Cartesian positions (in kpc)
        and the next three elements are the velocity components (in km/s).
    numpy.ndarray
        1D array of length `num_particles` representing the masses of the particles (in Msun).
    """

    ### MASS ###

    if np.isscalar(masses):
        masses = np.full(num_particles, masses)
    elif len(masses) != num_particles:
        raise ValueError("Length of masses array must be 1, or match the number of particles.")

    ### POSITION ###

    # Generate random positions uniformly within the spherical shell
    radii = np.random.uniform(inner_radius**3, outer_radius**3, num_particles)**(1/3)
    phi = np.random.uniform(0, 2 * np.pi, num_particles)
    cos_theta = np.random.uniform(-1, 1, num_particles)
    theta = np.arccos(cos_theta)
    positions = np.vstack((radii * np.sin(theta) * np.cos(phi),
                           radii * np.sin(theta) * np.sin(phi),
                           radii * np.cos(theta))).T

    # Initialize velocities
    velocity_magnitudes = infall_velocity_li(num_particles, virial_velocity, **kwargs)
    if np.any(velocity_magnitudes < 0):
        raise ValueError("Invalid velocity magnitudes: negative values detected.")
        
    # Generate random velocity directions
    phi_v = np.random.uniform(0, 2 * np.pi, num_particles)
    cos_theta_v = np.random.uniform(-1, 1, num_particles)
    theta_v = np.arccos(cos_theta_v)
    
    # Stack velocities into array
    velocities = velocity_magnitudes * np.vstack((
        np.sin(theta_v) * np.cos(phi_v), 
        np.sin(theta_v) * np.sin(phi_v), 
        np.cos(theta_v)
        )).T
    
    # Ensure radial velocity is inward
    radial_velocities = np.sum(velocities * positions, axis=1) # Unscaled radial velocity
    outward_mask = radial_velocities > 0
    velocities[outward_mask] *= -1

    return np.hstack((positions, velocities)), masses

def infall_velocity_li(n_particles, virial_velocity, mu1=1.20, sig1=0.20):
    """
    Generates an array of velocity magnitudes based on a log-normal distribution.
    The distribution parameters are best-fit values from Li+20
    (http://adsabs.harvard.edu/abs/2020arXiv200805710L)

    Parameters:
    -----------
    n_particles : int
        Number of velocity magnitudes to generate.
    v_vir : float
        Virial velocity of the host halo.
    mu1 : float, optional (default=1.20)
        Mean of the log-normal distribution in log-space.
    sig1 : float, optional (default=0.20)
        Standard deviation of the log-normal distribution in log-space.

    Returns:
    --------
    velocities : numpy.ndarray
        Array of velocity magnitudes, each multiplied by the virial velocity.
    """
    u = np.random.lognormal(mean=np.log(mu1), sigma=sig1, size=n_particles)
    return u * virial_velocity

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
        print(f"\n\nThe file {kwargs['out']} has been deleted to avoid overwrite error.\n")
    else:
        print(f"\n\nThe file {kwargs['out']} does not yet exist. Proceeding with command.\n")
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

def estimate_memory(n_particles, tstop, step):
    '''
    Estimate storage needs of output for gyrfalcON command

    Parameters:
    ----------
    n_particles : int
        Number of particles in the simulation.
    tstop : float
        Total integration time (in simulation units).
    step : float
        Time between outputs of simulation.

    Returns:
    -------
    mem : float
        Estimated memory of output file in MB
    '''
    n = int(tstop / step)
    # Each particle has 7 attributes, and assuming a double-precision float (8 bytes) for each attribute.
    bytes_per_particle = 7 * 8
    total_bytes = n_particles * bytes_per_particle * n
    
    return total_bytes / 1e6

def write_gyrfalcon_command(script_filename, server='mbp', chatter=True, **kwargs):
    """
    This function sets up a Bash script to run the gyrfalcON N-body simulation code and writes the script to a specified file.

    Parameters:
    ----------
    script_filename: str
        The name of the file where the Bash script will be written.
    server: str, optional
        The server where the script will run ('mbp' for local MacBook, 'astro' for remote).
        Defaults to 'mbp'.
    kwargs: dict
        Dictionary of keyword arguments for the gyrfalcON command. This includes parameters such as input file, output file, 
        and other simulation settings.
    """
    if 'in' not in kwargs:
        kwargs['in'] = kwargs.pop('in_', '-')

    step = kwargs.get('step')
    tstop = kwargs.get('tstop')
    
    val = " ".join([f"{k}={v}" for k, v in kwargs.items()])
    cmd = f"gyrfalcON {val}"

    # Determine shell type and NEMO source location based on server
    if server == 'astro':
        shell = "/bin/bash"
        nemo_source = "/home/ymt7/opt/nemo/nemo_start.sh"
    elif server == 'mbp':
        shell = "/bin/zsh"
        nemo_source = "/Users/yaronetokayer/nemo/nemo_start.sh"
    else:
        raise ValueError("Invalid server. Choose 'mbp' or 'astro'.")

    # Calculate estimated memory for different numbers of particles
    memory_examples = [
        (1e5, estimate_memory(1e5, tstop, step)),
        (2e5, estimate_memory(2e5, tstop, step)),
        (5e5, estimate_memory(5e5, tstop, step)),
        (1e6, estimate_memory(1e6, tstop, step)),
    ]

    # Create the memory estimation table as a string
    memory_table = "\n".join([f"{int(n):>7} particles : {mem:>10.2f} MB" for n, mem in memory_examples])

    # Script template with memory estimation
    script_content = dedent(f"""
    #!{shell}

    source {nemo_source}

    echo "Estimated storage needs for different particle counts:"
    echo "----------------------------------------------"
    echo "{memory_table}"
    echo "----------------------------------------------"

    read -p "Would you like to proceed? [Y/n] " -n 1 -r
    echo    # move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        if [ -f "{kwargs['out']}" ]; then
            rm "{kwargs['out']}"
            echo "The file {kwargs['out']} has been deleted to avoid overwrite error."
        else
            echo "The file {kwargs['out']} does not yet exist. Proceeding with command."
        fi

        # Start timing
        date1=$(date +"%s")

        # Start a background process to display the elapsed time
        (
          while true; do
            now=$(date +"%s")
            elapsed=$(($now - $date1))
            minutes=$(($elapsed / 60))
            seconds=$(($elapsed % 60))
            printf "\\rElapsed time: %02d:%02d" $minutes $seconds
            sleep 1
          done
        ) &
        timer_pid=$!

        {cmd}
        exit_code=$?

        # End timing
        date2=$(date +"%s")
        kill $timer_pid > /dev/null 2>&1

        # Calculate elapsed time
        diff=$(($date2-$date1))

        printf "\\r"  # Clear the elapsed time line
        echo "gyrfalcON command exited with code: $exit_code"
        echo "Simulation completed in $(($diff / 60)) minutes and $(($diff % 60)) seconds."

        # If the command fails, exit with an error code
        if [ $exit_code -ne 0 ]; then
            echo "Error: gyrfalcON command failed."
            exit $exit_code
        fi
    else
        echo "Simulation aborted by user."
    fi
    """)

    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)

    if chatter:
        print(f"Script written to {script_filename}")

# def write_gyrfalcon_command(script_filename, **kwargs):
#     """
#     This function sets up a Bash script to run the gyrfalcON N-body simulation code and writes the script to a specified file.

#     Parameters:
#     ----------
#     script_filename: str
#         The name of the file where the Bash script will be written.
#     kwargs: dict
#         Dictionary of keyword arguments for the gyrfalcON command. This includes parameters such as input file, output file, 
#         and other simulation settings.
#     """
#     if 'in' not in kwargs:
#         kwargs['in'] = kwargs.pop('in_', '-')

#     step = kwargs.get('step')
#     tstop = kwargs.get('tstop')
    
#     val = " ".join([f"{k}={v}" for k, v in kwargs.items()])
#     cmd = f"gyrfalcON {val}"

#     # Calculate estimated memory for different numbers of particles
#     memory_examples = [
#         (1e5, estimate_memory(1e5, tstop, step)),
#         (2e5, estimate_memory(2e5, tstop, step)),
#         (5e5, estimate_memory(5e5, tstop, step)),
#         (1e6, estimate_memory(1e6, tstop, step)),
#     ]

#     # Create the memory estimation table as a string
#     memory_table = "\n".join([f"{int(n):>7} particles : {mem:>10.2f} MB" for n, mem in memory_examples])

#     # Script template with memory estimation
#     script_content = dedent(f"""
#     #!/bin/bash

#     source /home/ymt7/opt/nemo/nemo_start.sh

#     echo "Estimated storage needs for different particle counts:"
#     echo "----------------------------------------------"
#     echo "{memory_table}"
#     echo "----------------------------------------------"

#     read -p "Would you like to proceed? [Y/n] " -n 1 -r
#     echo    # move to a new line
#     if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
#         if [ -f "{kwargs['out']}" ]; then
#             rm "{kwargs['out']}"
#             echo "The file {kwargs['out']} has been deleted to avoid overwrite error."
#         else
#             echo "The file {kwargs['out']} does not yet exist. Proceeding with command."
#         fi

#         {cmd}
#     else
#         echo "Simulation aborted by user."
#     fi
#     """)

#     with open(script_filename, 'w') as script_file:
#         script_file.write(script_content)

#     print(f"Script written to {script_filename}")

# def generate_spherical_shell_old(inner_radius, outer_radius, num_particles, virial_velocity, masses, sig1=0.20):
#     """
#     DEPRICATED!! NEW FUNCTIONS, ONE FOR OUR METHOD, ONE FOR ZZ METHOD
#     Generates particle data (positions, velocities, and masses) for a spherical shell of particles.
#     The particles are uniformly distributed in space within the spherical shell, and their velocities
#     are random and isotropic with magnitudes drawn from a specified probability distribution.

#     Parameters:
#     ----------
#     inner_radius : float
#         Inner radius of the spherical shell (in kpc).
#     outer_radius : float
#         Outer radius of the spherical shell (in kpc).
#     num_particles : int
#         Number of particles to generate.
#     virial_velocity : float
#         Virial velocity of the host halo (in km/s), to be passed to the velocity magnitude function.
#     masses : float or numpy.ndarray
#         Masses of the particles in units of Msun. If a single number is provided, all particles will have the same mass.
#         If an array is provided, it should have length `num_particles`.
#     sig1 : float, optional (default=0.20)
#         Standard deviation of the log-normal distribution in log-space for the velocity distribution.

#     Returns:
#     -------
#     numpy.ndarray
#         (n, 6) array where the first three elements of each row are the Cartesian positions (in kpc)
#         and the next three elements are the velocity components (in km/s).
#     numpy.ndarray
#         1D array of length `n` representing the masses of the particles (in Msun).
#     """

#     # Generate random positions uniformly within the spherical shell
#     radii = np.random.uniform(inner_radius**3, outer_radius**3, num_particles)**(1/3)
#     phi = np.random.uniform(0, 2 * np.pi, num_particles)
#     cos_theta = np.random.uniform(-1, 1, num_particles)
#     theta = np.arccos(cos_theta)

#     # Convert spherical coordinates to Cartesian coordinates
#     x = radii * np.sin(theta) * np.cos(phi)
#     y = radii * np.sin(theta) * np.sin(phi)
#     z = radii * np.cos(theta)
#     positions = np.vstack((x, y, z)).T

#     # Placeholder for velocity magnitudes
#     velocity_magnitudes = infall_velocity(num_particles, virial_velocity, sig1=sig1)

#     # Generate random isotropic velocity directions
#     phi_v = np.random.uniform(0, 2 * np.pi, num_particles)
#     cos_theta_v = np.random.uniform(-1, 1, num_particles)
#     theta_v = np.arccos(cos_theta_v)

#     # Convert spherical velocity components to Cartesian coordinates
#     vx = velocity_magnitudes * np.sin(theta_v) * np.cos(phi_v)
#     vy = velocity_magnitudes * np.sin(theta_v) * np.sin(phi_v)
#     vz = velocity_magnitudes * np.cos(theta_v)
#     velocities = np.vstack((vx, vy, vz)).T

#     # Handle mass input
#     if np.isscalar(masses):
#         masses = np.full(num_particles, masses)
#     elif len(masses) != num_particles:
#         raise ValueError("Length of masses array must match the number of particles.")

#     # Combine positions and velocities into one array
#     particle_data = np.hstack((positions, velocities))

#     return particle_data, masses
