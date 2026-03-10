## put main optimization function here

import numpy as np
from src.freewake_parse import freewake_input, freewake_run
from src import materials
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


def material_costs(mat):

    if mat == 'pla':
        material = materials.pla()
    elif mat == 'xps':
        material = materials.xps()
    elif mat == 'aluminum_6061':
        material = materials.aluminum_6061()
    elif mat == 'carbon':
        material = materials.carbon()

    price = material.price

    return price


def material_density(mat):

    if mat == 'pla':
        material = materials.pla()
    elif mat == 'xps':
        material = materials.xps()
    elif mat == 'aluminum_6061':
        material = materials.aluminum_6061()
    elif mat == 'carbon':
        material = materials.carbon()

    density = material.rho

    return density


def battery(parallel, series):

    # 21700 cell parameters
    cap_cell = 5000
    eta_batt = 0.88
    cost_per_cell = 2.75 # USD/cell
    mass_per_cell = 0.069 # kg

    # constant wiring mass
    wiring_mass = 0.1 # kg

    # nominal voltage
    V_nom = series*3.7 # V

    # battery pack energy
    energy = (parallel*cap_cell*eta_batt*V_nom)/1000

    # battery pack mass
    batt_mass = (parallel*series*mass_per_cell) + wiring_mass

    # battery pack cost
    batt_price = parallel*series*cost_per_cell

    return energy, batt_price, batt_mass


def mass(spar_mat, skin_mat, spar_volume, skin_volume):
    # CONSTANT fuse-empennage mass
    mass_fuse = 2.5 # kg

    # CONSTANT battery pack mass
    _, _, mass_batt = battery(6,6)

    # spar mass
    spar_density = material_density(spar_mat)
    mass_spar = spar_volume*spar_density

    # skin mass
    skin_density = material_density(skin_mat)
    mass_skin = skin_volume*skin_density

    total_mass = mass_spar + mass_skin + mass_batt + mass_fuse

    return total_mass


def wing_area(wingspan, mid_chord, tip_chord):

    # CONSTANT root chord
    root_chord = 0.15

    S = (wingspan/4)*((2*mid_chord) + root_chord + tip_chord)

    return S


def power_eqn(V, a, b, c):
    return np.array(a*(V**2) + b*V + c)


def aero(wingspan, mid_chord, tip_chord, root_twist, mid_twist, tip_twist, spar_mat, skin_mat, spar_volume, skin_volume, deflect_tip, deflect_mid):

    # CONSTANTS
    g = 9.81

    # Aircraft sizing
    S = wing_area(wingspan, mid_chord, tip_chord)
    AR = (wingspan**2)/S

    # Aircraft weight
    m = mass(spar_mat, skin_mat, spar_volume, skin_volume)
    W = m*g

    # Create input file for FreeWake
    freewake_input(wingspan, mid_chord, tip_chord, root_twist, mid_twist, tip_twist, W, deflect_tip, deflect_mid)
    df_perf, _ = freewake_run()

    # Fit second-order curve to airspeed vs. power
    coefficients, _ = curve_fit(power_eqn, df_perf['Vinf'], df_perf['Preq'])
    a_fit, b_fit, c_fit = coefficients

    # Find speed for max range
    V_maxR = np.sqrt(c_fit/a_fit)

    # # Find power required at speed for max range
    Preq = power_eqn(V_maxR, a_fit, b_fit, c_fit)

    # Find aoa for max range
    # CL = W/(0.5*1.225*(V_maxR**2)*wingspan)
    # CL_alpha = (2*np.pi)*(AR/(AR+2))
    # aoa_maxR = np.rad2deg(CL/CL_alpha)
    f_aoa_max = interp1d(df_perf['Vinf'],df_perf['alpha'],kind='linear')
    aoa_maxR = f_aoa_max(V_maxR)

    freewake_input(wingspan, mid_chord, tip_chord, root_twist, mid_twist, tip_twist, W, deflect_tip, deflect_mid, aoa_maxR, aoa_maxR, 1)
    _, df_load = freewake_run(aoa_maxR)

    y_pos = df_load['yo']
    y_load = 0.5*1.225*V_maxR*V_maxR*df_load['S']*df_load['cl']

    return V_maxR, Preq, y_pos, y_load
    

def price(spar_mat, skin_mat, spar_volume, skin_volume):
    # total price as a function of materials, spar volume, and skin volume

    # CONSTANT cost of fuselage, empennage, avionics
    price_fuse = 500 # USD

    # CONSTANT cost of battery pack
    _, price_batt, _ = battery(6,6)
    
    # cost of spar
    spar_cost = material_costs(spar_mat)
    spar_density = material_density(spar_mat)
    price_spar = spar_cost*spar_volume*spar_density

    # cost of skin
    skin_cost = material_costs(skin_mat)
    skin_density = material_density(skin_mat)
    price_skin = skin_cost*skin_volume*skin_density

    total_price = price_spar + price_skin + price_batt + price_fuse
    
    return total_price


def range(wingspan, mid_chord, tip_chord, root_twist, mid_twist, tip_twist, spar_mat, skin_mat, spar_volume, skin_volume):
    # CONSTANTS
    eta_p = 0.7 # powertrain efficiency

    # Battery parameters
    E_batt, _, _ = battery(6,6)

    # Aero
    V_maxR, Preq = aero(wingspan, mid_chord, tip_chord, root_twist, mid_twist, tip_twist, spar_mat, skin_mat, spar_volume, skin_volume)

    R = V_maxR*((E_batt*eta_p)/Preq)

    return R


def cost_func(wingspan, mid_chord, tip_chord, cap_area, t_skin, root_twist, mid_twist, tip_twist, skin_mat, spar_mat):
    # cost function as a function of all 9 design variables (10 because I assumed skin and spar materials are independent)

    # Structures model here with inputs of cap_area and skin thickness, outputs of spar volume and skin volume
    # This is just an estimate, needs to be more detailed
    # flange width, flange thickness, 
    spar_volume = cap_area*wingspan
    S = wing_area(wingspan, mid_chord, tip_chord)
    skin_volume = 2*S*t_skin

    # Range model (with aero model)
    range_est = range(wingspan, mid_chord, tip_chord, root_twist, mid_twist, tip_twist, spar_mat, skin_mat, spar_volume, skin_volume)

    # Price model
    price_est = price(spar_mat, skin_mat, spar_volume, skin_volume)

    # Minimize this function (maximizes range, minimizes price, equally weighted)
    # help function be a convex function
    # divide range by first estimate, divide cost by first estimate, use initial run
    costs = (-range_est/first_range) + (price_est/first_price)

    return costs


