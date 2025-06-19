import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.integrate import solve_ivp
import csv
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from CRUST import CRUST
# Constants
r_sat = 2.7e14  # g/cm^3
c = 2.998e8  # m/s
MeV_to_J = 1.60218e-13  # 1 MeV in Joules


# Conversion function
def conv_to_MeV(value):
    result = value * 1e3  # to kg/m^3
    result = result * c ** 2  # to J/m^3
    result = result / MeV_to_J  # to MeV/m^3
    result = result * 1e-45  # to MeV/fm^3
    return result


# HLPS_2 function
def HLPS_2(P):
    return 172.858 * (1 - np.exp(-P / 22.8644)) + 2777.75 * (1 - np.exp(-P / 1909.97)) + 161.553
def HLPS_3(P):
    return 131.811 * (1 - np.exp(-P/4.41577)) + 924.143 * (1 - np.exp(- P/523.736)) + 81.5682

# EOSPath class
class EOSPath:
    def __init__(self, P0, E0, segment_densities, gammas, name="EOS"):
        self.segment_densities = np.array(segment_densities)
        self.gammas = np.array(gammas)
        self.P0 = P0
        self.E0 = E0
        self.name = name
        self.Pi = []
        self.Ei = []
        self.K_list = []
        self.linear_start = None
        self.linear_eps = None

    def calculate_K(self, P, rho, gamma):
        return P / rho ** gamma

    def calculate_P(self, rho, K, gamma):
        return K * rho ** gamma

    def calculate_E(self, P, Ki, gamma, rho_i, P_i, E_i):
        if gamma == 1:
            return (E_i / rho_i) * P / Ki + np.log(1 / rho_i) * P - P * np.log(Ki / P)
        else:
            return (E_i / rho_i - P_i / (rho_i * (gamma - 1))) * (P / Ki) ** (1 / gamma) + P / (gamma - 1)

    def dE_dP(self, P, Ki, gamma, rho_i, P_i, E_i):
        if gamma == 1:
            return E_i / (rho_i * Ki) + np.log(1 / rho_i) + 1 - np.log(Ki / P)
        else:
            return (1 / (Ki * gamma)) * (E_i / rho_i - P_i / (rho_i * (gamma - 1))) * (P / Ki) ** (1 / gamma - 1) + 1 / (gamma - 1)

    def EOS_Linear(self, P, P_transition, E_transition):
        
        return E_transition + (P - P_transition)
    
    def calculate_parameters(self):
        # Initialize storage lists
        Pi_vals = [self.P0]
        Ei_vals = [self.E0]
        Ki_vals = []

        n = len(self.gammas)

        # Calculate Pi, Ei, Ki arrays for each segment
        for i in range(n):
            gamma = self.gammas[i]
            rho_i = self.segment_densities[i]
            rho_next = self.segment_densities[i + 1]

            K_i = self.calculate_K(Pi_vals[i], rho_i, gamma)
            Ki_vals.append(K_i)

            P_next = self.calculate_P(rho_next, K_i, gamma)
            Ei_next = self.calculate_E(P_next, K_i, gamma, rho_i, Pi_vals[i], Ei_vals[i])

            Pi_vals.append(P_next)
            Ei_vals.append(Ei_next)

        # Fine pressure sampling for slope checking
        P_range = np.linspace(Pi_vals[0], Pi_vals[-1], 10_000)
        P_tr = None
        E_tr = None

        # Function to compute slope dE/dP for piecewise polytropic EOS at arbitrary P
        def slope_at_P(P):
            # Find which segment P belongs to
            for i in range(len(Pi_vals) - 1):
                if Pi_vals[i] <= P <= Pi_vals[i + 1]:
                    return self.dE_dP(P, Ki_vals[i], self.gammas[i], self.segment_densities[i], Pi_vals[i], Ei_vals[i])
            # If beyond known segments, assume linear EOS slope = 1
            if self.linear_start and P > self.linear_start:
                return 1
            return 1  # fallback

        # Find the first pressure where slope < 1 (causality violation)
        for P in P_range:
            slope = slope_at_P(P)
            if slope < 1:
                P_tr = P
                # Calculate corresponding energy at transition
                for i in range(len(Pi_vals) - 1):
                    if Pi_vals[i] <= P_tr <= Pi_vals[i + 1]:
                        E_tr = self.calculate_E(P_tr, Ki_vals[i], self.gammas[i], self.segment_densities[i], Pi_vals[i], Ei_vals[i])
                        break
                # Truncate lists at transition
                for idx, P_bound in enumerate(Pi_vals):
                    if P_bound >= P_tr:
                        Pi_vals = Pi_vals[:idx + 1]
                        Ei_vals = Ei_vals[:idx + 1]
                        Pi_vals[-1] = P_tr
                        Ei_vals[-1] = E_tr
                        break
                break

        # Save results to the object
        self.Pi = Pi_vals
        self.Ei = Ei_vals
        self.K_list = Ki_vals
        self.linear_start = P_tr
        self.linear_eps = E_tr

    def get_energy_from_pressure(self, P):
        for i in range(len(self.Pi) - 1):
            if self.Pi[i] <= P <= self.Pi[i + 1]:
                return self.calculate_E(P, self.K_list[i], self.gammas[i], self.segment_densities[i], self.Pi[i], self.Ei[i])
        if P > self.Pi[-1]:
            if self.linear_start is None or self.linear_eps is None:
                # fallback if linear start/eps not set
                return self.Ei[-1]  # or some other reasonable fallback
            return self.linear_eps + (P - self.linear_start)

       

# TOV equation
def tov_rhs(r, z, eos_object):
    M, P = z
    if P <= 0:
        return [0, 0]
    elif  0.184 <= P <= 2.816:
        epsilon = HLPS_2(P)
    elif P > 2.816:
        # eos = EOS(P)
        epsilon = eos_object.get_energy_from_pressure(P)
    else:
        crust = CRUST(P)
        epsilon = crust.equation()
    # epsilon = eos_object.get_energy_from_pressure(P)
    dM_dr = 11.2e-6 * r ** 2 * epsilon
    dP_dr = -1.474 * (epsilon * M / r ** 2) * (1 + P / epsilon) * (1 + 11.2e-6 * r ** 3 * P / M) * (1 - 2.948 * M / r) ** (-1)
    return [dM_dr, dP_dr]
# Stopping event
def stop_when_pressure_small(r, y):
    return y[1] - 1e-10
stop_when_pressure_small.terminal = True
stop_when_pressure_small.direction = -1
# Solve one TOV star
def solve_star(P_central, eos_object):
    z0 = [1e-12, P_central]
    r_span = (1e-6, 2e6)
    sol = solve_ivp(lambda r, z: tov_rhs(r, z, eos_object),
                    r_span,
                    z0,
                    method='RK45',
                    atol=1e-10,
                    rtol=1e-8,
                    events=stop_when_pressure_small,
                    max_step=0.01)
    return sol

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

def get_segment_rhos(gamma_1, num_segments):
    ρ_sat_MeV = conv_to_MeV(r_sat)
    if gamma_1 <= 2:
        ρ_high = 16 * ρ_sat_MeV
    elif 2 < gamma_1 <= 3:
        ρ_high = 12 * ρ_sat_MeV
    else:
        ρ_high = 9 * ρ_sat_MeV

    log_ρ0 = np.log(ρ_sat_MeV)
    log_ρn = np.log(7.5 * ρ_sat_MeV)
    segment_rhos = np.exp(np.linspace(log_ρ0, log_ρn, num_segments + 1))
    segment_rhos[-1] = ρ_high
    return segment_rhos

if __name__ == "__main__":
    P_sat = 2.816 # in MeV/fm^3
    E_sat = HLPS_3(P_sat)
    segments = 5
    gamma_options = [1, 2, 3, 4]

    all_gamma_paths = list(product(gamma_options, repeat=segments))
    eos_objects = []

    for idx, gamma_path in enumerate(all_gamma_paths):
        gamma_1 = gamma_path[0]
        segment_rhos = get_segment_rhos(gamma_1, segments)
        eos = EOSPath(P_sat, E_sat, segment_rhos, gamma_path, name=f"EOS_{idx+1}")
        eos.calculate_parameters()
        eos_objects.append(eos)

    jobs = []
    for eos in eos_objects:
        gamma_1 = eos.gammas[0]
        P_final = eos.Pi[-1]

        ic1 = np.arange(1.8, 5, 0.1)
        if P_final > 700:
            ic2 = np.arange(5, 3500, 1)
        else:
            ic2 = np.arange(5, P_final, 1)

        initial_pressures = np.concatenate((ic1, ic2), axis=None)
        jobs.append((eos.name, eos.P0, eos.E0, eos.segment_densities, eos.gammas, initial_pressures))

    os.makedirs('TOV_results_1_2_3_4_HLPS_3', exist_ok=True)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_model, job) for job in jobs]
        for future in as_completed(futures):
            model_name, model_results = future.result()
            with open(f"TOV_results_1_2_3_4_HLPS_3/{model_name}_TOV.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Mass", "Radius", "Pressure", "Type"])
                for m, r, p in model_results:
                    writer.writerow([m, r, p, 0])