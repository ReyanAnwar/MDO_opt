import numpy as np
import materials

# Airfoil thickness ratio (MH32 ≈ 8.7% @ 30% chord)
tc_ratio = 0.087

# gravity
g = 9.81

# MATERIAL PROPERTIES
def get_material(mat):
    if mat == 'pla':
        return materials.pla()
    elif mat == 'xps':
        return materials.xps()
    elif mat == 'aluminum_6061':
        return materials.aluminum_6061()
    elif mat == 'carbon':
        return materials.carbon()
    else:
        raise ValueError(f"Unknown material: {mat}")

# CHORD DISTRIBUTION
def chord_dist(y, span, root_chord, mid_chord, tip_chord):
    half_span = span / 2
    y_mid = half_span / 2
    if y <= y_mid:
        return root_chord + (mid_chord-root_chord)*(y/y_mid)
    else:
        return mid_chord + (tip_chord - mid_chord)*((y - y_mid) / (half_span - y_mid))

# SHEAR AND BENDING MOMENT
def shear_moment(y, lift):
    N = len(y)
    dy = y[1] - y[0]
    V = np.zeros(N)
    M = np.zeros(N)
    for i in reversed(range(N-1)):
        V[i] = V[i+1] + lift[i]*dy
        M[i] = M[i+1] + V[i]*dy
    return V, M

# CAP AREA REQUIRED
def cap_area_required(M, h, sigma_allow):
    A_cap = M/(sigma_allow*h)
    return max(A_cap, 1e-6)

# WEB THICKNESS REQUIRED
def web_thickness_required(V, h, mat):
    material = get_material(mat)
    tau_allow = 0.6 * material.sigma_allow
    t_web = V/(tau_allow*h)
    return max(t_web, 0.01*h)

# MOMENT OF INERTIA
def inertia(A_cap, h):
    return 2*A_cap*(h/2)**2

# STRUCTURAL SOLVER
def solve_structure(span,
                    root_chord,
                    mid_chord,
                    tip_chord,
                    t_skin_root,
                    t_skin_mid,
                    t_skin_tip,
                    spar_mat,
                    skin_mat,
                    b_flange,
                    t_flange,
                    t_web,
                    y_span,
                    lift_distribution):

    N = len(y_span)
    dy = y_span[1] - y_span[0]

    # Interpolate skin thickness along span
    t_skin = np.interp(y_span, [0, span/4, span/2], [t_skin_root, t_skin_mid, t_skin_tip])

    # Shear and moment
    V, M = shear_moment(y_span, lift_distribution)

    material = get_material(spar_mat)
    sigma_allow = material.sigma_allow
    E = material.E

    EI = np.zeros(N)
    max_stress = 0
    spar_volume = 0
    skin_volume = 0

    for i in range(N):
        y = y_span[i]
        c = chord_dist(y, span, root_chord, mid_chord, tip_chord)

        # Internal spar height from airfoil thickness minus skin
        h = tc_ratio*c - 2*t_skin[i]
        h = max(h, 1e-4)

        # Use flange and web dimensions from optimizer
        A_cap = b_flange * t_flange

        I = 2*A_cap*(h/2)**2
        EI[i] = E*I

        sigma = M[i]/(A_cap*h)
        max_stress = max(max_stress, sigma)

        spar_area = 2*A_cap + t_web*h
        spar_volume += spar_area*dy
        skin_volume += 2*c*t_skin[i]*dy

    # DEFLECTION CALCULATION
    curvature = M / EI
    slope = np.zeros(N)
    deflection = np.zeros(N)
    for i in range(1, N):
        slope[i] = slope[i-1] + curvature[i]*dy
        deflection[i] = deflection[i-1] + slope[i]*dy

    mid_index = int(N/2)
    midspan_deflection = deflection[mid_index]
    tip_deflection = deflection[-1]

    return midspan_deflection, tip_deflection, spar_volume, skin_volume, max_stress