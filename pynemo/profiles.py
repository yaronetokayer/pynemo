from astropy.cosmology import FlatLambdaCDM
from astropy.constants import G
from astropy import units as u
import numpy as np
import agama

def truncNFW_prof(m_in, c, tau=2.0, z=0.0, cosmo=FlatLambdaCDM(H0=70, Om0=0.3), mass_kind='mvir', Delta_vir=97.0, cutoffStrength=5):
    """
    Make a 'truncated NFW' profile specified by the virial mass, concentration parameter, and truncation radius

    Arguments
    ---------
    m_in : float or Quantity
        If mass_kind='mvir': M(<rvir) in Msun.
        If mass_kind='total': total mass (Msun) of the truncated profile.
    c : float
        Concentration (rvir / rs).
    tau : float, optional
        Truncation radius in units of rvir.
        Default is 2.
    z: float
        redshift (deafult is 0)
    cosmo: astropy cosmology object
    mass_kind : {'mvir','total'} or bool
        If True, treat m_in as total mass. If False, as mvir.
        (String values preferred.)
    Delta_vir : float, optional
        Virial overdensity.
        Default is 200.
    cutoffStrength : float, optional
        cutoffStrength in Agama Potential constructor
        Default is 5

    Returns
    -------
    pot : agama.Potential
    param : dict
        Includes rs, rvir, rcut, c, tau, z, mvir, mtot, mass_param (Agama 'mass').
    """

    # normalize inputs
    if isinstance(mass_kind, bool):
        mass_kind = 'total' if mass_kind else 'mvir'
    if not isinstance(m_in, u.Quantity):
        m_in = m_in * u.Msun

    # scale-free fraction f(c,tau) = M(<rvir)/M_total
    f = _fraction_mvir_over_total(c, tau, cutoffStrength=cutoffStrength)  # dimensionless

    if mass_kind == 'mvir':
        mvir = m_in.to(u.Msun)
        mtot = (mvir.value / f) * u.Msun
    elif mass_kind == 'total':
        mtot = m_in.to(u.Msun)
        mvir = (f * mtot.value) * u.Msun        
    else:
        raise ValueError("mass_kind must be 'mvir', 'total', or bool.")
    mass_param = mtot.value

    # physical rvir from the (implied) mvir
    rvir = rvir_nfw(mvir, z=z, cosmo=cosmo, 
                    Delta_vir=Delta_vir)            # kpc (float)
    rs   = rvir / c                                 # kpc
    rcut = tau * rvir                               # kpc

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
        rvir=rvir, rs=rs, rcut=rcut, c=c, tau=tau, z=z,
        mvir=mvir.to_value(u.Msun), mtot=mtot.to_value(u.Msun),
        mass_kind=mass_kind, mass_param=mass_param
    ))

    return pot, meta

def mvir_nfw(rvir, z=0, cosmo=FlatLambdaCDM(H0=70, Om0=0.3), Delta_vir=200):
    """
    3D mass of an NFW halo contained within a radius of rvir.
    This is equivalent to passing rvir to m_nfw_3d, but the equation simplifies at rvir.
    See, e.g., Wright and Brainerd (2000)
    
    Inputs:
    rvir - array-like, radius of the halo inside which the mass density is Delta_vir*rho_c
           astropy units expected
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=70, Om0=0.3)
    Delta_vir : float, optional
        Virial overdensity.
        Default is 200.
    
    Returns:
    mvir - mass enclosed within rvir (Msol)
    """
    
    h = cosmo.H(z) # Hubble parameter
    
    rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    return ( ( 4 * Delta_vir * np.pi / 3 ) * rho_c * rvir**3 ).to(u.Msun)

def rvir_nfw(mvir, z=0, cosmo=FlatLambdaCDM(H0=70, Om0=0.3), Delta_vir=200):
    """
    rvir of an NFW halo given mvir.
    See, e.g., Wright and Brainerd (2000)
    
    Inputs:
    mvir - array-like, mass within the radius of the halo inside which the mass density is Delta_vir*rho_c
           astropy units expected
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=70, Om0=0.3)
    Delta_vir : float, optional
        Virial overdensity.
        Default is 200.
    
    Returns:
    rvir - rvir (kpc)
    """
    # OLD VERSION
    # h = cosmo.H(z) # Hubble parameter
    
    # rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    # return ( ( ( 3 * mvir ) / ( 800 * np.pi * rho_c ) )**(1/3) ).to(u.kpc)

    mvir = u.Quantity(mvir, u.Msun)

    rho_c = cosmo.critical_density(z).to(u.Msun/u.kpc**3)

    rvir = ((3 * mvir) / (4 * np.pi * Delta_vir * rho_c))**(1/3)

    return rvir.to(u.kpc)

def _fraction_mvir_over_total(c, tau, cutoffStrength=5):
    """
    Scale-free fraction f(c,tau) = M(<rvir) / M_total
    computed once using rvir=1, rs=1/c, rcut=tau (mass=1).
    """
    base = dict(type='Spheroid', mass=1.0, scaleRadius=1.0/c,
                alpha=1, beta=3, gamma=1, outerCutoffRadius=tau, cutoffStrength=cutoffStrength)
    pot = agama.Potential(**base)
    return pot.enclosedMass(1.0)  # since total mass=1