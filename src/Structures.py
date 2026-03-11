import numpy as np
import materials
import Mass

# Airfoil thickness ratio (MH32 ≈ 8.7% @ 30% chord)
tc_ratio = 0.087

# gravity
g = 9.81

# MATERIAL PROPERTIES (I think this should call the information from the main)
def get_material(mat):

    if mat == 'pla':
        m = materials.pla()
    elif mat == 'xps':
        m = materials.xps()
    elif mat == 'aluminum_6061':
        m = materials.aluminum_6061()
    elif mat == 'carbon':
        m = materials.carbon()
    else:
        raise ValueError(f"Unkown material: {mat}")

    return m

# CHORD DISTRIBUTION
def chord_dist(y, span, root_chord, mid_chord, tip_chord):

    half_span = span / 2
    y_mid = half_span / 2

    if y <= y_mid:
        # straight section
        c = root_chord + (mid_chord-root_chord)*(y/y_mid) # Midchord - Rootchord = 0 -> this can be simplified to c = rootchord later (Remember REYAN)
    else:
        # tapered section
        c = mid_chord + (tip_chord - mid_chord)*((y - y_mid) / (half_span - y_mid))

    return c

# SHEAR AND MOMENT (this should calculate it across the span)
def shear_moment(y, lift):

    N = len(y)

    dy = y[1] - y[0]

    V = np.zeros(N)
    M = np.zeros(N)

    for i in reversed(range(N-1)):

        V[i] = V[i+1] + lift[i]*dy
        M[i] = M[i+1] + V[i]*dy

    return V, M

# CAP AREA (Ideal I-beam approximation)
def cap_area_required(M, h, sigma_allow):

    A_cap = M/(sigma_allow*h)

    A_cap = max(A_cap, 1e-6)

    return A_cap

# WEB THICKNESS (This should prevent the web from becoming TOO thin)
def web_thickness_required(V, h, mat):

    material = get_material(mat)

    # tau_allow = 0.6*material.sigma_y
    tau_allow = 0.6*material.sigma_allow

    t_web = V/(tau_allow*h)

    return max(t_web,0.01*h)

# MOMENT OF INERTIA
def inertia(A_cap, h):

    I = 2*A_cap*(h/2)**2

    return I

# STRUCTURAL SOLVER
# Inputs
def solve_structure(span,
                    root_chord,
                    mid_chord,
                    tip_chord,
                    t_skin,
                    spar_mat,
                    skin_mat,
                    lift_distribution):
    
    lift_distribution = lift_distribution[:19]

    half_span = span/2

    N = len(lift_distribution)

    y_span = np.linspace(0,half_span,N)

    dy = y_span[1] - y_span[0]

    # PRELIMINARY VOLUME ESTIMATION (Initial Guess)
    spar_volume = 0
    skin_volume = 0

    for i in range(N):

        y = y_span[i]

        c = chord_dist(y,span,root_chord,mid_chord,tip_chord)

        h = tc_ratio*c

        A_cap_guess = 1e-5
        t_web_guess = 0.01*h

        spar_area = 2*A_cap_guess + t_web_guess*h

        spar_volume += spar_area*dy

        skin_area = 2*c

        skin_volume += skin_area*t_skin*dy

    # CALL MASS SOLVER (I think I did this right)
    total_mass = Mass.total_mass(
                    spar_volume,
                    skin_volume,
                    spar_mat,
                    skin_mat
                )

    weight = total_mass*g

    # SHEAR AND MOMENT
    V,M = shear_moment(y_span,lift_distribution)

    # SPAR SIZING
    material = get_material(spar_mat)

    sigma_allow = material.sigma_allow
    E = material.E

    EI = np.zeros(N)

    spar_volume = 0
    skin_volume = 0

    max_stress = 0

    for i in range(N):

        y = y_span[i]

        c = chord_dist(y,span,root_chord,mid_chord,tip_chord)

        h = tc_ratio*c

        A_cap = cap_area_required(M[i],h,sigma_allow)

        t_web = web_thickness_required(V[i],h,spar_mat)

        I = inertia(A_cap,h)

        EI[i] = E*I

        sigma = M[i]/(A_cap*h)

        max_stress = max(max_stress,sigma)

        spar_area = 2*A_cap + t_web*h

        spar_volume += spar_area*dy

        skin_area = 2*c

        skin_volume += skin_area*t_skin*dy

    # DEFLECTIONS
    curvature = M/EI

    slope = np.zeros(N)
    deflection = np.zeros(N)

    for i in range(1,N):

        slope[i] = slope[i-1] + curvature[i]*dy

        deflection[i] = deflection[i-1] + slope[i]*dy


    mid_index = int(N/2)

    midspan_deflection = deflection[mid_index]

    tip_deflection = deflection[-1]

    #Outputs
    return (midspan_deflection,
            tip_deflection,
            spar_volume,
            skin_volume,
            max_stress,
            total_mass)