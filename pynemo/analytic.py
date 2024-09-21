import numpy as np

def f_hern(e, m, a, g):
    """
    Analytic isotropic distribution function f(E) for a spherically symmetric Hernquist profile
    See Hernquist (1990) 10.1086/168845

    Parameters:
    ----------
    e : float or numpy.ndarray
        Energy at which to compute f(E).
    m : float
        Total 3D integrated mass of the profile.
    a : float
        Scale length of the profile.
    g : float
        Gravitational constant consistent with units used for all other inputs.

    Returns:
    -------
    float or numpy.ndarray
        f(E) corresponding to e
    """
    q = np.sqrt(-a*e/g/m)
    v_g = np.sqrt(g*m/a)

    f_e = (
        m / ( 8 * np.sqrt(2) * np.pi**3 * a**3 * v_g**3 )
        * 1 / ( 1 - q**2 )**(5/2)
        * (
            3 * np.arcsin(q) 
            + q * np.sqrt(1 - q**2) * (1 - 2*q**2) * (8 * q**4 - 8 *q**2 - 3)
        )
    )
    return f_e

def g_hern(e, m, a, g):
    """
    Analytic isotropic density of states function g(E) for a spherically symmetric Hernquist profile
    See Hernquist (1990) 10.1086/168845


    Parameters:
    ----------
    e : float or numpy.ndarray
        Energy at which to compute g(E).
    m : float
        Total 3D integrated mass of the profile.
    a : float
        Scale length of the profile.
    g : float
        Gravitational constant consistent with units used for all other inputs.

    Returns:
    -------
    float or numpy.ndarray
        g(E) corresponding to e
    """
    q = np.sqrt(-a*e/g/m)
    v_g = np.sqrt(g*m/a)

    g_e = (
        2 * np.sqrt(2) * np.pi**2 * a**3 * v_g / 3 / q**5
        * (
            3 * (8*q**4 - 4*q**2 + 1) * np.arccos(q) 
            - q*np.sqrt(1 - q**2) * (4*q**2 - 1) * (2*q**2 + 3)
        )
    )
    return g_e