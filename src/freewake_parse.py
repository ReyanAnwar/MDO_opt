# Script to create an input file for Freewake 2018, based on SW117 aircraft
# Inputs: wingspan, wing area, aircraft weight
import subprocess
import pandas as pd
import os
import math

def freewake_input(span, chord_mid, chord_tip, twist_mid, twist_tip, weight, deflect_tip=0, deflect_mid=0, alpha_min=0.5, alpha_max=10, alpha_delta=0.5):

    chord_root = 0.15
    area = (((chord_mid+chord_tip)/2)*(span/4) + ((chord_mid+chord_root)/2)*(span/4))*2
    wing_y_quart = span/4
    wing_y_half = span/2
    cmac = chord_root

    wing_x_root = 0.000
    wing_x_mid = (chord_root*0.303)-(chord_mid*0.303)
    wing_x_tip = (chord_root*0.303)-(chord_tip*0.303)

    # HARDCODED: filepath of Freewake input file
    path = r"C:\Users\mayar\Documents\Ryerson\Grad Classes\AE8139 MDO\MDO_opt\src\fw\input.txt"

    with open(path, "w") as f:
        f.write("Input file for FreeWake 2014 \nFormat will not work with older versions \nPlease note that the program uses equal, number and : signs as special recognizers! \nThe results are written to the sub-directory output' \nInput file for kite foil, normal wing-tail config\n")

        f.write("\nRelaxed wake (yes 1, no 0): relax = 0 \n")
        f.write("Steady (1) or unsteady (2): aerodynamics = 1 \n")
        f.write("Viscous solutions (1) or inviscid (0) viscous = 1 \n")
        f.write("Symmetrical geometry (yes 1, no 0): sym = 1 \n")
        f.write("Longitudinal trim (yes 1, no 0): trim = 0 (yes means m has to be 1) \n \n")


        f.write("Max. number of time steps:	maxtime = 3 \n")
        f.write("Width of each time step (sec):	deltime = 100.1 \n")
        f.write("Convergence delta-span effic.:	deltae = 0.00 (0 if onlytimestepping) \n \n")


        f.write("Freestream velocity (leave value 1): Uinf = 1.0 \n")
        f.write(f"AOA beginning, end, step size [deg]: alpha	= {alpha_min:.2f} {alpha_max:.2f} {alpha_delta} \n")
        f.write("Sideslip angle [deg]: beta	= 0.0 \n")
        f.write("Density: density =	1.225 \n")
        f.write("Kinematic viscosity: nu = 1.4600000e-05 \n \n")


        f.write(f"Reference area: S = {area} \n")
        f.write(f"Reference span: b = {span} \n \n")


        f.write(f"Mean aerodynamic chord: cmac = {cmac} \n")
        f.write(f"Aircraft weight (N): W = {weight} \n")
        f.write("CG location (x y z): cg = 0.0525 0.000000000 0.000000000 \n")
        f.write("CMo of wing: CMo = 0.000000000 \n \n")


        f.write("No. of wings (max. 5): wings = 1 \n")
        f.write("No. of panels:	panels = 2 \n")
        f.write("No. of chordwise lifting lines: m = 1 \n")
        f.write("No. of airfoils (max. 15):	airfoils = 10 \n \n")

        f.write("Panel boundary conditions:\n")
        f.write("Symmetry line - 	10\n")
        f.write("Between panels - 	220\n")
        f.write("Free end - 		100\n \n")


        f.write("Panel #:1. Number of spanwise elements (n) = 2 \n")
        f.write("Neighbouring panels (0 for none) left: 0 right: 2 \n")
        f.write("xleft	        yleft	        zleft	        chord	        epsilon	    Bound.Cond. Airfoil \n")
        f.write(f"{wing_x_root}	0.000	        0.00	        {chord_root}	0.00    	10			1 \n")
        f.write("xleft	        yleft	        zleft	        chord	        epsilon	    Bound.Cond. Airfoil \n")
        f.write(f"{wing_x_mid}	{wing_y_quart}	{deflect_mid}   {chord_mid}		{twist_mid}	220			1 \n \n")

        f.write("Panel #:2. Number of spanwise elements (n) = 2 \n")
        f.write("Neighbouring panels (0 for none) left: 1 right: 0 \n")
        f.write("xleft	        yleft	        zleft	        chord	        epsilon	    Bound.Cond. Airfoil \n")
        f.write(f"{wing_x_mid}	{wing_y_quart}	{deflect_mid}	{chord_mid}		{twist_mid}	220			1 \n")
        f.write("xleft	        yleft	        zleft	        chord	        epsilon	    Bound.Cond. Airfoil \n")
        f.write(f"{wing_x_tip}	{wing_y_half}	{deflect_tip}	{chord_tip}	    {twist_tip}	100			1 \n \n")

        # f.write("Panel #:3. Number of spanwise elements (n) = 2 \n")
        # f.write("Neighbouring panels (0 for none) left: 2 right: 4 \n")
        # f.write("xleft	        yleft	        zleft	        chord	        epsilon	    Bound.Cond. Airfoil \n")
        # f.write(f"{wing_x_root}	0.000	        0.000	        {chord_root}	{twist_root}    220			1 \n")
        # f.write("xleft	        yleft	        zleft	        chord	        epsilon	    Bound.Cond. Airfoil \n")
        # f.write(f"{wing_x_mid}	{wing_y_quart}	{deflect_mid}   {chord_mid}		{twist_mid}	220			1 \n \n")

        # f.write("Panel #:4. Number of spanwise elements (n) = 2 \n")
        # f.write("Neighbouring panels (0 for none) left: 3 right: 0 \n")
        # f.write("xleft	        yleft	        zleft	        chord	        epsilon	    Bound.Cond. Airfoil \n")
        # f.write(f"{wing_x_mid}	{wing_y_quart}	{deflect_mid}   {chord_mid}		{twist_mid}	220			1 \n")
        # f.write("xleft	        yleft	        zleft	        chord	        epsilon	    Bound.Cond. Airfoil \n")
        # f.write(f"{wing_x_tip}	{wing_y_half}	{deflect_tip}	{chord_tip}		{twist_tip}	100			1 \n \n")

        f.write("Tail area is 0 m^2 \n \n")


        f.write("%<- special identifier \n")
        f.write("Vertical tail information: \n")
        f.write("Number of panels (max 5) = 0 \n")
        f.write("no. chord area	airfoil \n")
        # f.write("no. chord area airfoil \n")
        # f.write("1 0.4625 0.0636 10 \n")
        # f.write("2 0.3875 0.0533 10 \n")
        # f.write("3 0.3125 0.0430 10 \n")
        # f.write("4 0.2375 0.0327 10 \n \n")


        f.write("Fuselage information: \n")
        f.write("Number of sections (max 20) = 0 \n")
        f.write("Width of each section =	0\n")
        f.write("Panel where transition occurs =	0\n")
        f.write("No.	Diamter\n")
        f.write("1	0\n \n")

        f.write("Interference drag = 0.0% \n")
        f.write("############## \n")
    
    return

def freewake_run(aoa=None):

    # HARDCODED: filepath of Freewake executable
    aoa_col = ['index','xo','yo','zo','cn','cl','cy','cd','A','B','C','S','span','chord','nu','epsilon','psi','phiLE','#']
    fw_folder = r"C:\Users\mayar\Documents\Ryerson\Grad Classes\AE8139 MDO\MDO_opt\src\fw"
    fw_exe = "fw_2025.exe"
    result_check = subprocess.run([fw_exe], cwd=fw_folder, capture_output=True, text=True, shell=True)
    # print("Freewake ran successfully if 0: \n", result_check.returncode)

    # Get output Performance.txt file
    perf_output_path = r"C:\Users\mayar\Documents\Ryerson\Grad Classes\AE8139 MDO\MDO_opt\src\fw\output\Performance.txt"
    df_performance = pd.read_csv(perf_output_path, sep=r'\s+', skiprows=3)

    if aoa is not None:
        filename = f"AOA{aoa:.2f}.txt"
        force_base_path = r"C:\Users\mayar\Documents\Ryerson\Grad Classes\AE8139 MDO\MDO_opt\src\fw\output"
        force_output_path = os.path.join(force_base_path,filename)
        df_force = pd.read_csv(force_output_path, skiprows=4, sep=r'\s+', nrows=8, header=None, names=aoa_col)
    else:
        df_force = []

    return df_performance, df_force


