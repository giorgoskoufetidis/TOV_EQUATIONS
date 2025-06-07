import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Constants
r_sat = 2.7e14  # g/cm^3
c = 2.998e8
MeV_to_J = 1.60218e-13

# Helper conversion
def conv_to_MeV(value):
    result = value * 1e3  # kg/m^3
    result = result * c**2  # J/m^3
    result = result / MeV_to_J  # MeV/m^3
    result = result * 1e-45  # MeV/fm^3
    return result

def HLPS_2(P):
    return 172.858 * (1 - np.exp(- P/22.8644)) + 2777.75 * (1 - np.exp(- P/1909.97)) + 161.553

# EOS Class
class EOSPath:
    def __init__(self, P0, E0, segment_densities, gammas, name= "EOS"):
        self.segment_densities = np.array(segment_densities)
        self.gammas = np.array(gammas)
        self.P0 = P0
        self.E0 = E0
        self.Pi = []
        self.Ei = []
        self.K_list = []
        self.name = name 
    def calculate_K(self, P, rho, gamma):
        return P / rho**gamma

    def calculate_P(self, rho, K, gamma):
        return K * rho**gamma

    def calculate_E(self, P, Ki, gamma, rho_i, P_i, E_i):
        if gamma == 1:
            return (E_i / rho_i) * P / Ki + np.log(1 / rho_i) * P - P * np.log(Ki / P)
        else:
            return (E_i / rho_i - P_i / (rho_i * (gamma - 1))) * (P / Ki)**(1 / gamma) + P / (gamma - 1)

    def calculate_parameters(self):
        Pi = [self.P0]
        Ei = [self.E0]
        K_list = []

        for i in range(len(self.gammas)):
            gamma = self.gammas[i]
            rho_i = self.segment_densities[i]
            P_i = Pi[-1]
            E_i = Ei[-1]

            K_i = self.calculate_K(P_i, rho_i, gamma)
            K_list.append(K_i)

            rho_next = self.segment_densities[i + 1]
            P_next = self.calculate_P(rho_next, K_i, gamma)
            E_next = self.calculate_E(P_next, K_i, gamma, rho_i, P_i, E_i)

            Pi.append(P_next)
            Ei.append(E_next)

        self.Pi = Pi
        self.Ei = Ei
        self.K_list = K_list
        return Pi, Ei, K_list

    def get_energy_from_pressure(self, P_query):
    # Find the correct segment for interpolation
        for i in range(len(self.Pi) - 1):
            if self.Pi[i] <= P_query <= self.Pi[i + 1]:
                P1 = self.Pi[i]
                E1 = self.Ei[i]
                K1 = self.K_list[i]
                gamma = self.gammas[i]
                return self.calculate_E(P_query, K1, gamma, self.segment_densities[i], P1, E1)

        # Extrapolate below the first segment
        if P_query < self.Pi[0]:
            P1 = self.Pi[0]
            E1 = self.Ei[0]
            K1 = self.K_list[0]
            gamma = self.gammas[0]
            return self.calculate_E(P_query, K1, gamma, self.segment_densities[0], P1, E1)

        # Extrapolate beyond the last segment
        if P_query > self.Pi[-1]:
            P1 = self.Pi[-2]
            E1 = self.Ei[-2]
            K1 = self.K_list[-1]
            gamma = self.gammas[-1]
            return self.calculate_E(P_query, K1, gamma, self.segment_densities[-2], P1, E1)

            

# Setup: Ensure all EOS paths start from the same pressure and energy density
P_sat = 1.722  # Starting pressure
E_sat = HLPS_2(P_sat)  # Starting energy density
r_sat_Mev_fm3 = conv_to_MeV(r_sat)  # Initial density in MeV/fm³

segments = 6
gamma_options = [1, 2, 3, 4]
segment_rhos = np.exp(np.linspace(np.log(r_sat_Mev_fm3), np.log(r_sat_Mev_fm3 * segments), segments + 1))
all_gamma_paths = list(product(gamma_options, repeat=segments))  # 81 combinations

# Define pressure array P (for example)
P_array = np.linspace(P_sat, 1500, 2000)  # Adjust size for desired resolution

# Plot energy densities for each EOS path
plt.figure(figsize=(10, 7))
colors = plt.cm.viridis(np.linspace(0, 1, len(all_gamma_paths)))
eos_objects = []

# Plot all EOS paths ensuring they start from the same point
# Mask P_array to start at P_sat
P_array_masked = P_array[P_array >= P_sat]

# Plot all EOS paths ensuring they start from the same point
for idx, gamma_path in enumerate(all_gamma_paths[120:]):
    eos = EOSPath(P_sat, E_sat, segment_rhos, gamma_path, name=f"EOS {idx+1}")
    eos_objects.append(eos)
    

# from polytropic_stars.EOSPath import EOSPath
import matplotlib.pyplot as plt
from CRUST import CRUST
import numpy as np
from scipy.integrate import solve_ivp
import csv
import os 
from multiprocessing import Process
# TOV right-hand side equations
from concurrent.futures import ProcessPoolExecutor, as_completed

# TOV right-hand side equations
def tov_rhs(r, z, eos_object):
    M, P = z
    if P <= 0:
        return [0, 0]

    if  0.184 <= P <= 1.722:
        epsilon = HLPS_2(P)
    elif P > 1.722:
        # eos = EOS(P)
        epsilon = eos_object.get_energy_from_pressure(P)
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
def process_model(args):
    model_name, P0, E0, segment_densities, gammas, initial_pressures = args
    
    eos_object = EOSPath(P0, E0, segment_densities, gammas, name=model_name)
    eos_object.calculate_parameters()
    
    results = []
    for P_center in initial_pressures:
        sol = solve_star(P_center, eos_object)
        if sol.status == 1:
            R_star = sol.t_events[0][0]
            M_star = sol.y_events[0][0][0]
            results.append((M_star, R_star, P_center))
            print(f"Model:{model_name} | Mass:{M_star:.4f} | Radius:{R_star:.4f} | Pc={P_center:.2f}")
        else:
            print(f"Warning: Model {model_name} Pc={P_center:.2f} did not converge.")
    return model_name, results


if __name__ == "__main__":
    # Parameters and EOS objects setup
    P_sat = 1.722
    E_sat = HLPS_2(P_sat)  
    segments = 6
    gamma_options = [1, 3 ,5]
    
    r_sat_Mev_fm3 =conv_to_MeV(r_sat) 
    segment_rhos = np.exp(np.linspace(np.log(r_sat_Mev_fm3), np.log(r_sat_Mev_fm3 * segments), segments + 1))
    from itertools import product
    all_gamma_paths = list(product(gamma_options, repeat=segments))

    eos_objects = []
    for idx, gamma_path in enumerate(all_gamma_paths):  # Just 3 for quick test
        eos = EOSPath(P_sat, E_sat, segment_rhos, gamma_path, name=f"EOS_{idx+1}")
        eos.calculate_parameters()
        eos_objects.append(eos)

    ic1 = np.arange(2, 5, 0.1)
    ic2 = np.arange(5, 1501, 1)
    initial_pressures = np.concatenate((ic1, ic2), axis=None)

    jobs = [(eos.name, eos.P0, eos.E0, eos.segment_densities, eos.gammas, initial_pressures) for eos in eos_objects[:30]]

    os.makedirs('TOV_results_test', exist_ok=True)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_model, job) for job in jobs]

        for future in as_completed(futures):
            model_name, model_results = future.result()

            with open(f"TOV_results_test/{model_name}_TOV.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Mass", "Radius", "Pressure", "Type"])
                for m, r, p in model_results:
                    writer.writerow([m, r, p, 0])