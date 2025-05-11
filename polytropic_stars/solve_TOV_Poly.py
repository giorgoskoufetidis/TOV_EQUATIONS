
# from polytropic_stars.EOSPath import EOSPath
from EOSGenerator import EOSGenerator
import matplotlib.pyplot as plt
import random
from CRUST import CRUST
import numpy as np
from scipy.integrate import solve_ivp
import random
import csv
import os 
from EOS import EOS
# TOV right-hand side equations

# TOV right-hand side equations
def tov_rhs(r, z, eos_object):
    M, P = z

    if P <= 0:
        return [0, 0]

    if P >= 0.184:
        eos = EOS(P)
        epsilon = eos.models()[10]
    else:
        crust = CRUST(P)
        epsilon = crust.equation()

    dM_dr = 11.2e-6 * r**2 * epsilon
    dP_dr = -1.474 * (epsilon * M / r**2) * (1 + P / epsilon) * (1 + 11.2e-6 * r**3 * P / M) * (1 - 2.948 * M / r)**(-1)

    return [dM_dr, dP_dr]

# Stopping event when pressure becomes tiny
def stop_when_pressure_small(r, y):
    return y[1] - 1e-10  # Stop when P ~ 0
stop_when_pressure_small.terminal = True
stop_when_pressure_small.direction = -1

# Solve a single TOV star
def solve_star(P_central, eos_object):
    z0 = [1e-12, P_central]  # Initial [M, P]
    r_span = (1e-6, 2e6)     # Integrate from tiny radius up to big radius
    sol = solve_ivp(lambda r, z: tov_rhs(r, z, eos_object),
                    r_span, 
                    z0,
                    method='RK45',
                    atol=1e-10,
                    rtol=1e-8,
                    events=[stop_when_pressure_small],
                    max_step=0.1)
    return sol

# Process one EOS model
def process_model(model_name, eos_object, initial_pressures):
    results = []

    for i, P_center in enumerate(initial_pressures):
        sol = solve_star(P_center, eos_object)

        if sol.status == 1:  # event triggered (good convergence)
            R_star = sol.t_events[0][0]  # surface radius
            M_star = sol.y_events[0][0][0]  # final mass
            results.append((M_star, R_star, P_center))
            print(f"Model:{model_name} | Mass:{M_star:.4f} | Radius:{R_star:.4f} | Pc={P_center:.2f}")
        else:
            print(f"Warning: Model {model_name} Pc={P_center:.2f} did not converge.")

    return model_name, results

# MAIN EXECUTION
if __name__ == "__main__":
    # Create folder for results
    os.makedirs('TOV_results', exist_ok=True)

    # Generate EOS models
    p_sat = 3.2  # MeV/fm³
    initial_rho = 0.16  # fm⁻³
    segments = 4
    gamma_options = [1, 5]

    generator = EOSGenerator(p_saturation=p_sat, initial_density=initial_rho, segments=segments, gamma_options=gamma_options)
    generator.generate()
    eos_models = generator.get_models()

    # Define central pressures
    ic1 = np.arange(1.5, 5, 0.1)
    ic2 = np.arange(5, 1201, 1)
    initial_pressures = np.concatenate((ic1, ic2), axis=None)

    # Solve for all EOS models
    for eos_model in eos_models:
        model_name = eos_model.name
        eos_object = eos_model

        model_name, model_results = process_model(model_name, eos_object, initial_pressures)

        # Save results
        with open(f"TOV_results/{model_name}_TOV.csv", "w", newline="") as f:
            write_obj = csv.writer(f)
            write_obj.writerow(["Mass", "Radius", "Pressure", "Type"])
            for m, r, p in model_results:
                write_obj.writerow([m, r, p, 0])