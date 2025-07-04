import os
import csv
import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing
from functools import partial

hbar_c = 197.326
c = 2.998e8
class QEOS:
    def __init__(self, ms, delta, B):
        self.ms = ms
        self.delta = delta
        self.B = B

    def alpha(self):
        return -self.ms**2 / 6 + 2 * self.delta**2 / 3

    def energy_density(self, P):
        a = self.alpha()
        m = np.sqrt(-3 * a + np.sqrt(4 / 3 * np.pi**2 * (self.B + P) * hbar_c**3 + 9 * a**2))
        epsilon = 3 * P + 4 * self.B - 9 * a * m**2 / (np.pi**2 * hbar_c**3)
        return epsilon

def tov_rhs(r, z, eos_model):
    M = z[0]  
    P = z[1]  
    y = z[2]    
    if P <= 0: 
        return [0, 0, 0]
    epsilon = eos_model.energy_density(P)
    dP_de = (eos_model.energy_density(P + 1e-8) - eos_model.energy_density(P - 1e-8)) / (2e-8)
    
    F =  (1-1.474 * 11.2 * ( 10 **(-6)) * (r** 2) * (epsilon - P)) * ((1-2.948 * M / r) ** (-1))
    # Assuming math module is imported and variables M, r, P, e, dd are defined
    J = 1.474 * 11.2 * (10**(-6)) * (r**2) * (5 * epsilon + 9 * P + (epsilon + P) / (1 / dP_de)) *  ((1-2.948 * M / r) **(-1)) - 6 * \
    ((1-2.948 * M / r) ** (-1)) - 4 * ((1.474 ** 2) * (M ** 2)/(r ** 2)) *  ((1 + 11.2 *  (10 **(-6)) * (r** 3) * (P / M)) ** 2) * ((1-2.948 * M / r) ** (-2))
    # epsilon = eos_object.get_energy_from_pressure(P)
    dM_dr = 11.2e-6 * r ** 2 * epsilon
    dP_dr = -1.474 * (epsilon * M / r ** 2) * (1 + P / epsilon) * (1 + 11.2e-6 * r ** 3 * P / M) * (1 - 2.948 * M / r) ** (-1)
    dyrdt = (-y * y - y * F - J) / r
    return [dM_dr, dP_dr, dyrdt]
# Stopping event
def compute_k2(beta, yR):
    k2 = (8 *(beta ** 5)/5) * ((1-2 * beta)** 2) * (2-yR +2 * beta * (yR - 1)) * (2 *
    beta * (6-3 * yR +3 * beta *  (5 * yR -8))+ 4 * beta ** 3 * (13-11 * yR + beta * (3 *
    yR - 2) + 2 * beta ** 2 * (1 + yR)) + 3 * ((1 - 2 * beta)** 2) * (2 - yR + 2 * beta * (yR - 1)) * np.log(1 - 2 * beta)) ** (-1)
    return k2

def stability_point(ms, mn, delta):
    conversion_factor = hbar_c**3
    return (-ms**2 * mn**2 / (12 * np.pi**2) + delta**2 * mn**2 / (3 * np.pi**2) + mn**4 / (108 * np.pi**2)) / conversion_factor

def process_model(args):
    model_name, eos_model, initial_pressures, output_folder = args
    results = []

    for P0 in initial_pressures:
        z0 = [1e-12, P0, 2]  
        rmin = 1e-8
        rmax = 0.01

        final_mass = None
        final_radius = None
        final_pressure = None

        while z0[1] > 1e-11:  
            rhs_with_model = partial(tov_rhs, eos_model=eos_model)
            res = solve_ivp(rhs_with_model, (rmin, rmax), z0, method='RK45', atol=1e-12, rtol=1e-8)

            if res.success and res.t.size > 0:
                last_idx = -1
                final_radius = res.t[last_idx]
                final_mass = res.y[0][last_idx]
                final_pressure = res.y[1][last_idx]
                yR_int = res.y[2][last_idx]
                beta = 1.474 * final_mass / final_radius
                epsilon_s = 4 * eos_model.B 
                yR_ext = yR_int - (epsilon_s / c**2) / (final_mass / (4 * np.pi * final_radius ** 3))
                k2 = compute_k2(beta, yR_ext)
                if final_pressure <= 0:
                    break

                z0 = [final_mass, final_pressure, yR_ext]
                rmin = final_radius
                rmax = rmin + 0.01
            else:
                break

        if final_mass is not None and final_radius is not None and final_pressure is not None and yR_ext is not None and k2 is not None:
            results.append((final_mass, final_radius, P0, k2, yR_ext))
            print(f"{model_name} -> M: {final_mass}, R: {final_radius}, P: {P0}, k2: {k2}, yR: {yR_ext}")

    # Write CSV file immediately after finishing this model
    output_path = os.path.join(output_folder, f"TOV_results_{model_name}.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mass", "Radius", "Pressure","k2", "yR", "Type"])
        for mass, radius, pressure, k2, yR in results:
            writer.writerow([mass, radius, pressure, k2, yR, 1])

    return model_name, len(results)  # Return minimal info just for logging

def run_parallel_processing():
    delta_values = list(range(50, 301, 4)) 
    ms = 95
    B_values = list(range(60, 240, 4)) 
    # delta_values = [50, 100]
    # ms = 95 
    # B_values = [60, 100]
    mn = 939  
    eos_models = []
    model_index = 1
    output_folder = "TOV_results_Qark_Stars_tidal"
    os.makedirs(output_folder, exist_ok=True)

    for b in B_values:
        for d in delta_values:
            if b < stability_point(ms, mn, d):  
                model_name = f"CFL_{model_index}"
                eos_model = QEOS(ms, d, b)
                initial_pressures = np.concatenate((np.arange(1.5, 5, 0.1), np.arange(5, 1501, 1)), axis=None)
                eos_models.append((model_name, eos_model, initial_pressures, output_folder))
                model_index += 1

    print(f"Total Models: {len(eos_models)}")

    with multiprocessing.Pool() as pool:
        for model_name, count in pool.imap_unordered(process_model, eos_models):
            print(f"Finished writing {model_name} with {count} results")

    print("Parallel Processing Completed!")

if __name__ == "__main__":
    run_parallel_processing()
