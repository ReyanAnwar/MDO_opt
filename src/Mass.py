import materials
import numpy as np
from scipy.integrate import quad

# CONSTANT MASSES
BATTERY_MASS = 2.584          # kg
FUSE_EMPENNAGE_MASS = 2.5     # kg

# MATERIAL DENSITY
def get_density(mat):
    """Return the density of a given material (kg/m^3)"""
    if mat == 'pla':
        material = materials.pla()
    elif mat == 'xps':
        material = materials.xps()
    elif mat == 'aluminum_6061':
        material = materials.aluminum_6061()
    elif mat == 'carbon':
        material = materials.carbon()
    else:
        raise ValueError(f"Unknown material: {mat}")
    return material.rho


def spar_area(chord, t_skin, t_flange, w_flange, t_web):
    """
    Calculate spar area at a given chord location for given structural dimensions

    Inputs:
        chord = chord at spanwise position (m)
        t_skin = skin thickness at spanwise position (m)
        t_flange = flange thickness at spanwise position (m)
        w_flange = flange width at spanwise position (m)
        t_web = web thickness at spanwise position (m)
    
    Returns:
        a_spar = spar area (m^3)
        
    """
    
    thickness = 0.303*chord
    h_web = thickness - (2*t_skin) - (2*t_flange)
    a_spar = (t_flange*w_flange*2) + (t_web*h_web)
    
    return a_spar

def struct_vol(wingspan, mid_chord, tip_chord, t_skin_root, t_skin_mid, t_skin_tip, t_flange, w_flange, t_web):
    """
    Calculate structural volume of spar and skin

    Input:
        wingspan = total wing span (m)
        mid_chord = chord at half of wing half-span (m)
        tip_chord = chord at wing tip (m)
        t_skin_root = skin thickness at wing root (m)
        t_skin_mid = skin thickness at half of wing half-span (m)
        t_skin_tip = skin thickness at wing tip (m)
        t_flange = flange thickness at wing root (m)
        w_flange = flange width at wing root (m)
        t_web = web thickness at wing root (m)

    Returns:
        v_spar = total spar volume (m^3)
        v_skin = total skin volume (m^3)

    """

    half_span = wingspan/2
    y_pos = np.linspace(0,half_span,100)

    a_y_spar = np.zeros(len(y_pos))
    chord_y = np.zeros(len(y_pos))
    thickness_y = np.zeros(len(y_pos))

    for i in range(len(y_pos)):
        if (y_pos[i] < half_span/2):
            y_local = y_pos[i]/(half_span/2)
            chord_y[i] = 0.15 + (mid_chord-0.15)*y_local
            thickness_y[i] = t_skin_root + (t_skin_mid-t_skin_root)*y_local
            t_flange_y = (t_flange/0.15)*chord_y[i]
            w_flange_y = (w_flange/0.15)*chord_y[i]
            t_web_y = (t_web/0.15)*chord_y[i]
            a_y_spar[i] = spar_area(chord_y[i], thickness_y[i], t_flange_y, w_flange_y, t_web_y)
        else:
            y_local = (y_pos[i]-(half_span/2))/(half_span/2)
            chord_y[i] = mid_chord + (tip_chord-mid_chord)*y_local
            thickness_y[i] = t_skin_mid + (t_skin_tip-t_skin_mid)*y_local
            t_flange_y = (t_flange/0.15)*chord_y[i]
            w_flange_y = (w_flange/0.15)*chord_y[i]
            t_web_y = (t_web/0.15)*chord_y[i]
            a_y_spar[i] = spar_area(chord_y[i], thickness_y[i], t_flange_y, w_flange_y, t_web_y)

    v_spar = (np.trapezoid(a_y_spar, y_pos))*2
    v_skin = (np.trapezoid(2*chord_y*thickness_y, y_pos))*2

    return v_spar, v_skin

# STRUCTURAL MASS
def structural_mass(spar_volume, skin_volume, spar_mat, skin_mat):
    """
    Calculate spar and skin masses given volumes and material types.

    Inputs:
        spar_volume: float, m^3
        skin_volume: float, m^3
        spar_mat: str, material name
        skin_mat: str, material name

    Returns:
        mass_spar, mass_skin: tuple of floats in kg
    """
    rho_spar = get_density(spar_mat)
    rho_skin = get_density(skin_mat)

    mass_spar = spar_volume * rho_spar
    mass_skin = skin_volume * rho_skin

    return mass_spar, mass_skin

# TOTAL AIRCRAFT MASS
def total_mass(spar_volume, skin_volume, spar_mat, skin_mat):
    """
    Calculate total mass of aircraft including battery and fuselage/empennage.

    Inputs:
        spar_volume: float, m^3
        skin_volume: float, m^3
        spar_mat: str
        skin_mat: str

    Returns:
        total_mass: float, kg
    """
    mass_spar, mass_skin = structural_mass(spar_volume, skin_volume, spar_mat, skin_mat)

    total = mass_spar + mass_skin + BATTERY_MASS + FUSE_EMPENNAGE_MASS
    return total