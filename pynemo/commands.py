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
