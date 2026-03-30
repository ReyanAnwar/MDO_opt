## put main optimization function here

import numpy as np
from freewake_parse import freewake_input, freewake_run
import materials
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from Mass import total_mass, struct_vol
from Structures import solve_structure
import math
import os
import shutil
import tempfile

def material_costs(mat):
    """
    Return material price given material name

    Inputs:
        mat = material name, string
    
    Returns:
        price = cost (USD) per kg of material
    """

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
    """
    Return material density given material name
    
    Inputs:
        mat = material name, string
    
    Returns:
        density = density of material (kg/m^3)
    """

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
    """
    Battery pack energy, price and mass for a given number of cells in parallel and in series,
    pack is designed for 21700 cells with a battery efficiency of 88%

    
    """

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


def aero(freewake_folder, weight, wingspan, mid_chord, tip_chord, mid_twist, tip_twist, deflect_tip, deflect_mid):

    fw_output_folder = os.path.join(freewake_folder,f"output")
    if os.path.exists(fw_output_folder):
        shutil.rmtree(fw_output_folder)
        os.mkdir(fw_output_folder)

    # Create input file for FreeWake
    freewake_input(freewake_folder, wingspan, mid_chord, tip_chord, mid_twist, tip_twist, weight, deflect_tip, deflect_mid)
    df_perf, _ = freewake_run(freewake_folder)

    # Fit second-order curve to airspeed vs. power-required
    df_clean = df_perf.dropna()
    if df_clean.empty or df_clean.shape[0]<4:
        V_maxR = 1
        Preq = 1e3
        y_pos = np.linspace((wingspan/16),(wingspan/2)-(wingspan/16),4)
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
            freewake_input(freewake_folder, wingspan, mid_chord, tip_chord, mid_twist, tip_twist, weight, deflect_tip, deflect_mid, aoa_maxR, aoa_maxR, 1)
            df_perf_aoa, df_load = freewake_run(freewake_folder,aoa_maxR)
            V_maxR = df_perf_aoa['Vinf'][0]
            Preq = df_perf_aoa['Preq'][0]
            y_pos = df_load['yo']
            y_load = 0.5*1.225*V_maxR*V_maxR*df_load['S']*df_load['cl']
        else:
            V_maxR = 1
            Preq = 1e3
            y_pos = np.linspace((wingspan/16),(wingspan/2)-(wingspan/16),4)
            y_load = np.ones(len(y_pos))*1e3     

    return V_maxR, Preq, y_pos, y_load
    

def price(spar_mat, skin_mat, spar_volume, skin_volume):
    """
    Calculates total price of aircraft as a function of volumes of materials used to construct wing spar
    and wing skin, each constructed with their respective materials

    Inputs:
        spar_mat = spar material, string
        skin_mat = skin material, string
        spar_volume = total volume of spar, m^3
        skin_volume = total volume of skin, m^3
    
    Returns:
        total_price = price of UAV including battery cost and fixed fuselage, empennage, and avionics cost, USD
    """
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
    """
    Range calculation given airspeed and power required, assumes a constant powertrain efficiency of 70% and
    a 6S6P battery pack made with 21700 cells each having a capacity of 5000mAh

    Inputs:
        V_maxR = airspeed, speed for max range (m/s)
        Preq = power required at the given airspeed (W)
    
    Returns:
        R = range, assuming zero wind (km)
    """
    eta_p = 0.7 # powertrain efficiency

    # Battery parameters
    E_batt, _, _ = battery(6,6)

    R = (V_maxR*((E_batt*eta_p)/Preq))*3.6 # km

    return R

def mat_index(i):
    """
    Select material given index value for optimizer

    Input:
        i = Discrete value between 0 and 3, corresponding to a material
    
    Returns:
        string of material name, used to identify material and properties in materials.py database
    """
    if i==0:
        return 'pla'
    if i==1:
        return 'xps'
    if i==2:
        return 'aluminum_6061'
    if i==3:
        return 'carbon'
    

def cost_func(wingspan, mid_chord, tip_chord, w_flange, t_flange, t_web, t_skin_root, t_skin_mid, t_skin_tip, mid_twist, tip_twist, skin_index, spar_index):

    """
    Solve minimizing cost function given values of design variables, to be used in an optimizer. Cost function minimizes cost and
    maximizes range for a given wing and structure design.

    Inputs:
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
        costs = output value of cost function
        range_est = range estimate for given UAV configuration
        mass_tot = total UAV mass
        deflect_tip = final tip deflection for given spar geometry and aerodynamic loads
    """

    # create a copy of freewake to run for this generation
    fw_source = r"C:\Users\mayar\Documents\Ryerson\Grad Classes\AE8139 MDO\MDO_opt\src\fw"
    # new_fw_folder = fw_source

    with tempfile.TemporaryDirectory() as temp_dir:
        fw_temp_folder = os.path.join(temp_dir, "case")
        new_fw_folder = shutil.copytree(fw_source, fw_temp_folder)

        # Initial run of range and cost functions with initial conitions of optimization variables
        # desired_price = 650.43
        # desired_range = 8.78
        desired_price = 600
        desired_range = 1000
        desired_mass = 10

        skin_mat = mat_index(skin_index)
        spar_mat = mat_index(spar_index)

        # Initial guess of skin and spar volumes for first run of aero model
        spar_volume, skin_volume = struct_vol(wingspan, mid_chord, tip_chord, t_skin_root, t_skin_mid, t_skin_tip, t_flange, w_flange, t_web)
        S = wing_area(wingspan, mid_chord, tip_chord)

        # Initial guess of weight
        mass_tot = total_mass(spar_volume, skin_volume, spar_mat, skin_mat)
        weight = mass_tot*9.81

        # First run of aero model assuming no tip deflection
        V_maxR, Preq, y_loc, y_load_old = aero(new_fw_folder, weight, wingspan, mid_chord, tip_chord, mid_twist, tip_twist, 0, 0)

        # print(f"Speed for max range: {V_maxR:.2f}")

        deflect_old = 0.00001
        deflection_delta = 1
        # print(f"Load Dist:{y_load_old}")
        i = 0

        # Loop between aero model and structure model until deflections converge
        # while (deflection_delta > 0.05):
        while (i<3):
            # fix the spar in this loop
            deflect_mid, deflect_tip, _, _, max_stress = solve_structure(wingspan, 0.15, mid_chord, tip_chord, t_skin_root, t_skin_mid, t_skin_tip, spar_mat, skin_mat, w_flange, t_flange, t_web, y_loc, y_load_old)

            # run aero model with new mass to get loading
            V_maxR, Preq, y_loc, y_load_new = aero(new_fw_folder, weight, wingspan, mid_chord, tip_chord, mid_twist, tip_twist, deflect_tip, deflect_mid)

            # deflection_delta = np.abs(deflect_tip - deflect_old)
            # try percent deflection
            deflection_delta = np.abs(deflect_tip-deflect_old)/deflect_old

            deflect_old = deflect_tip
            y_load_old = y_load_new

            # print(f"Tip deflection:{i}|{deflect_tip:.5f}")
            i = i+1

        # Range model (with aero model)
        range_est = range_km(V_maxR, Preq)
        # print(f"Range: {range_est:.2f} km")

        # Price model
        # price_est = price(spar_mat, skin_mat, spar_volume, skin_volume)
        # print(f"Price: ${price_est:.2f}")

        # Implement costraints
        penalty = 0
        if mass_tot > 10:
            penalty += ((mass_tot-10)/10)*10 # penalty as a ratio of how much over the limit mass is
        # if price_est > 1000:
        #     penalty += ((price_est-1000)/1000)*10 # penalty as a ratio of how much over the limit price is
        if deflect_tip > (0.15*(wingspan/2)):
            penalty += ((deflect_tip-(0.15*(wingspan/2)))/(0.15*(wingspan/2)))*20

        # Minimize this function (maximizes range, minimizes price, equally weighted)
        # For pygad, the fitness function must be maximized so negate cost
        # costs = (price_est/desired_price) - (range_est/desired_range) + penalty
        costs = (mass_tot/desired_mass) - (range_est/desired_range) + penalty

        shutil.rmtree(temp_dir)

    return costs, range_est, mass_tot, deflect_tip


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

def aero_CDi(mid_chord, tip_chord):
    # minimize cdi at a fixed cl, inviscid, 0.8
    wingspan = 5
    S = wing_area(wingspan, mid_chord, tip_chord)
    V_maxR, Preq, _, _ = aero(1, 5, mid_chord, tip_chord, 0, 0, 0, 0)

    
    # mass_penalty = max(0, mass_tot)

    costs = (1/Preq)

    return costs