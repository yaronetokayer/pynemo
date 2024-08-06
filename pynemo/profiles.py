from astropy.cosmology import FlatLambdaCDM
from astropy.constants import G
from astropy import units as u
import numpy as np
import agama

def truncNFW_prof(m200, c, tau=2, z=0, cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
    """
    Make a 'truncated NFW' profile specified by the virial mass, concentration parameter, and truncation radius

    Inputs:
    m200: astropy quantity
        radius of the halo inside which the mass density is 200*rho_c
    c: float
        concentration parameter
    tau: float
        ratio of truncation radius to virial radius (default is 2)
    z: float
        redshift (deafult is 0)
    cosmo: astropy cosmology object

    Returns:
    pot: agama potential object
    param: dict
        parameters of pot
        NOTE: "mass" includes total intergrated mass, which is greater than m200 passed into the function
              To get m200, use pot.enclosedMass(m200)
    """

    rvir = r200_nfw(m200, z=z, cosmo=cosmo)
    rs = rvir / c
    
    param = dict(
        type='Spheroid',
        mass=m200.value, scaleRadius=rs.value,
        alpha=1, beta=3, gamma=1,
        outerCutoffRadius=tau * rvir.value
    )
    pot = agama.Potential(**param)

    # Rescale to get correct mass within rvir
    param = dict(
        type='Spheroid',
        mass=m200.value * m200.value / pot.enclosedMass(rvir.value), 
        scaleRadius=rs.value,
        alpha=1, beta=3, gamma=1,
        outerCutoffRadius=tau * rvir.value
    )
    pot = agama.Potential(**param)

    return pot, param

def m200_nfw(r200, z=0, cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
    """
    3D mass of an NFW halo contained within a radius of r200.
    This is equivalent to passing r200 to m_nfw_3d, but the equation simplifies at r200.
    See, e.g., Wright and Brainerd (2000)
    
    Inputs:
    r200 - array-like, radius of the halo inside which the mass density is 200*rho_c
           astropy units expected
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=70, Om0=0.3)
    
    Returns:
    m200 - mass enclosed within r200 (Msol)
    """
    
    h = cosmo.H(z) # Hubble parameter
    
    rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    return ( ( 800 * np.pi / 3 ) * rho_c * r200**3 ).to(u.Msun)

def r200_nfw(m200, z=0, cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
    """
    r200 of an NFW halo given m200.
    See, e.g., Wright and Brainerd (2000)
    
    Inputs:
    m200 - array-like, mass within the radius of the halo inside which the mass density is 200*rho_c
           astropy units expected
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=70, Om0=0.3)
    
    Returns:
    r200 - r200 (kpc)
    """
    
    h = cosmo.H(z) # Hubble parameter
    
    rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    return ( ( ( 3 * m200 ) / ( 800 * np.pi * rho_c ) )**(1/3) ).to(u.kpc)