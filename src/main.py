## put main optimization function here

import numpy as np
from freewake_parse import freewake_input, freewake_run
import materials
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from Mass import total_mass, spar_vol
from Structures import solve_structure
import math


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


def wing_area(wingspan, mid_chord, tip_chord):

    # CONSTANT root chord
    root_chord = 0.15

    S = (wingspan/4)*((2*mid_chord) + root_chord + tip_chord)

    return S


def power_eqn(V, a, b, c):
    return np.array(a*(V**2) + b*V + c)


def aero(weight, wingspan, mid_chord, tip_chord, mid_twist, tip_twist, deflect_tip, deflect_mid):

    # Create input file for FreeWake
    freewake_input(wingspan, mid_chord, tip_chord, mid_twist, tip_twist, weight, deflect_tip, deflect_mid)
    df_perf, _ = freewake_run()

    # Fit second-order curve to airspeed vs. power
    df_clean = df_perf.dropna()
    if df_clean.empty or df_clean.shape[0]<3:
        V_maxR = 1
        Preq = 1e3
        y_pos = np.linspace(-(wingspan/2)+(wingspan/16),0-(wingspan/16),4)
        y_load = np.ones(len(y_pos))*1e3
    else:

        coefficients, _ = curve_fit(power_eqn, df_clean['Vinf'], df_clean['Preq'])
        a_fit, b_fit, c_fit = coefficients

        # Find speed for max range
        V_maxR = np.sqrt(c_fit/a_fit)

        # Find aoa for max range using interpolation
        f_aoa_max = interp1d(df_clean['Vinf'],df_clean['alpha'],kind='linear', fill_value='extrapolate')
        aoa_maxR = f_aoa_max(V_maxR)

        if not math.isnan(aoa_maxR):
            # Re-run FreeWake at aoa for max range
            freewake_input(wingspan, mid_chord, tip_chord, mid_twist, tip_twist, weight, deflect_tip, deflect_mid, aoa_maxR, aoa_maxR, 1)
            df_perf_aoa, df_load = freewake_run(aoa_maxR)
            V_maxR = df_perf_aoa['Vinf'][0]
            Preq = df_perf_aoa['Preq'][0]
            y_pos = df_load['yo']
            y_load = 0.5*1.225*V_maxR*V_maxR*df_load['S']*df_load['cl']
        else:
            V_maxR = 1
            Preq = 1e3
            y_pos = np.linspace(-(wingspan/2)+(wingspan/16),0-(wingspan/16),4)
            y_load = np.ones(len(y_pos))*1e3     

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

def range_km(V_maxR, Preq):
    # CONSTANTS
    eta_p = 0.7 # powertrain efficiency

    # Battery parameters
    E_batt, _, _ = battery(6,6)

    R = (V_maxR*((E_batt*eta_p)/Preq))*3.6 # km

    return R

def mat_index(i):
    if i==0:
        return 'pla'
    if i==1:
        return 'xps'
    if i==2:
        return 'aluminum_6061'
    if i==3:
        return 'carbon'
    

def cost_func(wingspan, mid_chord, tip_chord, w_flange, t_flange, t_web, t_skin_root, t_skin_mid, t_skin_tip, mid_twist, tip_twist, skin_index, spar_index):
    # cost function as a function of all 9 design variables (10 because I assumed skin and spar materials are independent)

    # Initial run of range and cost functions with initial conitions of optimization variables
    desired_range = 500
    desired_price = 1000

    skin_mat = mat_index(skin_index)
    spar_mat = mat_index(spar_index)

    # Initial guess of skin and spar volumes for first run of aero model
    spar_volume = spar_vol(wingspan, mid_chord, tip_chord, t_skin_root, t_skin_mid, t_skin_tip, t_flange, w_flange, t_web)
    skin_inbd_volume = t_skin_root*(((0.15+mid_chord)/2)*(wingspan/4))
    skin_outbd_volume = (t_skin_mid)*(((mid_chord+tip_chord)/2)*(wingspan/4))
    skin_volume = (skin_inbd_volume + skin_outbd_volume)*2
    S = wing_area(wingspan, mid_chord, tip_chord)

    # Initial guess of weight
    mass_tot = total_mass(spar_volume, skin_volume, spar_mat, skin_mat)
    weight = mass_tot*9.81

    # First run of aero model assuming no tip deflection
    V_maxR, Preq, y_loc, y_load_old = aero(weight, wingspan, mid_chord, tip_chord, mid_twist, tip_twist, 0, 0)
    deflect_old = 0
    deflection_delta = 1

    # Loop between aero model and structure model until deflections converge
    while (deflection_delta > 1e-5):
        # fix the spar in this loop
        deflect_mid, deflect_tip, spar_volume, skin_volume, max_stress = solve_structure(wingspan, 0.15, mid_chord, tip_chord, t_skin_root, t_skin_mid, t_skin_tip, spar_mat, skin_mat, w_flange, t_flange, t_web, y_loc, y_load_old)

        # run aero model with new mass to get loading
        V_maxR, Preq, y_loc, y_load_new = aero(weight, wingspan, mid_chord, tip_chord, mid_twist, tip_twist, deflect_tip, deflect_mid)

        deflection_delta = deflect_tip - deflect_old

        deflect_old = deflect_tip
        y_load_old = y_load_new

    # Range model (with aero model)
    range_est = range_km(V_maxR, Preq)

    # Price model
    price_est = price(spar_mat, skin_mat, spar_volume, skin_volume)

    # Implement costraints
    penalty = 0
    # if wing_load < load_cond:
    #     penalty = penalty + 10
    if mass_tot > 10:
        penalty += ((mass_tot-10)/10)*10
    if price_est > 1000:
        penalty += ((price_est-1000)/1000)*10
    if deflect_tip < (0.15*(wingspan/2)):
        penalty += ((deflect_tip-(0.15*(wingspan/2)))/(0.15*(wingspan/2)))*20

    # Minimize this function (maximizes range, minimizes price, equally weighted)
    # costs = (-range_est/desired_range) + (price_est/desired_cost)
    # For pygad, the fitness function must be maximized so swap signs
    costs = (price_est/desired_price) - (range_est/desired_range) + penalty

    # help function be a convex function
    # divide range by first estimate, divide cost by first estimate, use initial run

    return costs


def aero_cost(wingspan, mid_chord, tip_chord, mid_twist, tip_twist):

    t_skin = 0.001
    S = wing_area(wingspan, mid_chord, tip_chord)
    skin_volume = 2*S*t_skin
    mass_tot = total_mass(0, skin_volume, 'pla', 'pla')
    weight = mass_tot*9.81
    V_maxR, Preq, _, _ = aero(weight, wingspan, mid_chord, tip_chord, mid_twist, tip_twist, 0, 0)
    range_est = range_km(V_maxR, Preq)
    if mass_tot > 10:
        pen_mass = (mass_tot-10)*(-10)
    # if range_est < 0:
    #     pen_range = (range_est*10)
    
    # to be maximized
    costs = range_est 

    return costs

def aero_gradient_cost(wingspan, mid_chord, tip_chord, mid_twist, tip_twist):
    t_skin = 0.001
    S = wing_area(wingspan, mid_chord, tip_chord)
    skin_volume = 2*S*t_skin
    mass_tot = total_mass(0, skin_volume, 'pla', 'pla')
    weight = mass_tot*9.81
    V_maxR, Preq, _, _ = aero(weight, wingspan, mid_chord, tip_chord, mid_twist, tip_twist, 0, 0)
    range_est = range_km(V_maxR, Preq)
    if range_est <= 0:
        range_est = 1e6
    
    # mass_penalty = max(0, mass_tot)

    costs = (1/range_est)

    return costs