from astropy.cosmology import FlatLambdaCDM
from astropy.constants import G
from astropy import units as u
import numpy as np
import agama

def truncNFW_prof(m_in, c, tau=2.0, z=0.0, cosmo=FlatLambdaCDM(H0=70, Om0=0.3), mass_kind='m200'):
    """
    Make a 'truncated NFW' profile specified by the virial mass, concentration parameter, and truncation radius

    Arguments
    ---------
    m_in : float or Quantity
        If mass_kind='m200': M(<r200) in Msun.
        If mass_kind='total': total mass (Msun) of the truncated profile.
    c : float
        Concentration (r200 / rs).
    tau : float, optional
        Truncation radius in units of r200.
        Default is 2.
    z: float
        redshift (deafult is 0)
    cosmo: astropy cosmology object
    mass_kind : {'m200','total'} or bool
        If True, treat m_in as total mass. If False, as m200.
        (String values preferred.)

    Returns
    -------
    pot : agama.Potential
    param : dict
        Includes rs, r200, rcut, c, tau, z, m200, mtot, mass_param (Agama 'mass').
    """

    # normalize inputs
    if isinstance(mass_kind, bool):
        mass_kind = 'total' if mass_kind else 'm200'
    if not isinstance(m_in, u.Quantity):
        m_in = m_in * u.Msun

    # scale-free fraction f(c,tau) = M(<r200)/M_total
    f = _fraction_m200_over_total(c, tau)  # dimensionless

    if mass_kind == 'm200':
        m200 = m_in.to(u.Msun)
        mtot = (m200.value / f) * u.Msun
        mass_param = mtot.value
    elif mass_kind == 'total':
        mtot = m_in.to(u.Msun)
        m200 = (f * mtot.value) * u.Msun
        mass_param = mtot.value
    else:
        raise ValueError("mass_kind must be 'm200', 'total', or bool.")

    # physical r200 from the (implied) m200
    r200 = r200_nfw(m200, z=z, cosmo=cosmo)         # kpc (float)
    rs   = r200 / c                                 # kpc
    rcut = tau * r200                               # kpc

    # final, properly scaled potential
    param = dict(
        type='Spheroid',
        mass=mass_param,
        scaleRadius=rs.value,
        alpha=1, beta=3, gamma=1,
        outerCutoffRadius=rcut.value,
        cutoffStrength=5
    )
    pot = agama.Potential(**param)

    # package rich metadata
    meta = dict(param)
    meta.update(dict(
        r200=r200, rs=rs, rcut=rcut, c=c, tau=tau, z=z,
        m200=m200.to_value(u.Msun), mtot=mtot.to_value(u.Msun),
        mass_kind=mass_kind, mass_param=mass_param
    ))

    return pot, meta

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
    # OLD VERSION
    # h = cosmo.H(z) # Hubble parameter
    
    # rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    # return ( ( ( 3 * m200 ) / ( 800 * np.pi * rho_c ) )**(1/3) ).to(u.kpc)

    m200 = u.Quantity(m200, u.Msun)

    rho_c = cosmo.critical_density(z).to(u.Msun/u.kpc**3)

    r200 = ((3 * m200) / (4 * np.pi * 200 * rho_c))**(1/3)

    return r200.to(u.kpc)

def _fraction_m200_over_total(c, tau):
    """
    Scale-free fraction f(c,tau) = M(<r200) / M_total
    computed once using r200=1, rs=1/c, rcut=tau (mass=1).
    """
    base = dict(type='Spheroid', mass=1.0, scaleRadius=1.0/c,
                alpha=1, beta=3, gamma=1, outerCutoffRadius=tau, cutoffStrength=5)
    pot = agama.Potential(**base)
    return pot.enclosedMass(1.0)  # since total mass=1