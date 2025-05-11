from EOS import EOS 
from CRUST import CRUST
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
from multiprocessing import Pool
import csv

# Define TOV right-hand side equations
def tov_rhs(r, z, index):
    M = z[0]
    P = z[1]
    if P >= 0.184:
        eos = EOS(P)
        epsilon = eos.models()[index]
    else:
        crust = CRUST(P)
        epsilon = crust.equation()
    if P <= 0:
        return [0, 0]
    dM_dr = 11.2e-6 * r**2 * epsilon
    dP_dr = -1.474 * (epsilon * M / r**2) * (1 + P / epsilon) * (1 + 11.2e-6 * r**3 * P / M) * (1 - 2.948 * M / r)**(-1)
    return [dM_dr, dP_dr]

# Function to process a single model
def process_model(args):
    model_name, initial_conditions, model_index = args
    results = []
    ic = initial_conditions[model_name]
    for i in ic:
        z0 = [1e-12, i]
        P0 = i
        rmin = 1e-8
        rmax = 0.01

        Mf = np.array([])
        Pf = np.array([])
        R = np.array([])

        while P0 > 1e-12:
            rhs_with_index = partial(tov_rhs, index=model_index)
            res = solve_ivp(rhs_with_index, (rmin, rmax), z0, method='LSODA', atol=1e-12, rtol=1e-8)
            if res.success:
                R_step = res.t[~np.isnan(res.t)]
                Mf_step = res.y[0][~np.isnan(res.y[0])]
                Pf_step = res.y[1][~np.isnan(res.y[1])]

                if any(Pf_step < 0):
                    idx = np.argmax(Pf_step < 0)
                    R_step = R_step[:idx]
                    Mf_step = Mf_step[:idx]
                    Pf_step = Pf_step[:idx]

                R = np.append(R, R_step)
                Mf = np.append(Mf, Mf_step)
                Pf = np.append(Pf, Pf_step)

                z0 = [Mf[-1], Pf[-1]]
                rmin = R[-1]
                rmax = rmin + 0.001
                P0 = z0[1]
            else:
                break

        if len(R) > 0 and len(Mf) > 0:
            print(f"name:{model_name} m:{Mf[-1]} radius:{R[-1]} P0:{i}")
            results.append((Mf[-1], R[-1], i))

    return model_name, results

# Main execution
if __name__ == "__main__":
    models = [
        "MDI_1", "MDI_2", "MDI_3", "MDI_4", "NLD", "HHJ_1", "HHJ_2", "SKa",
        "SkI4", "HLPS_2", "HLPS_3", "SCVBB", "WFF_1", "WFF_2", "PS", "W", "BGP", "BL_1",
        "BL_2", "DH", "APR_1"
    ]

    ic1 = np.arange(1.5, 5, 0.1)
    ic2 = np.arange(5, 1201, 1)
    init = np.concatenate((ic1, ic2), axis=None)

    initial_conditions = {model: init for model in models}

    tasks = [(model_name, initial_conditions, i) for i, model_name in enumerate(models)]

    with Pool() as pool:
        results = pool.map(process_model, tasks)
    for model_name, model_results in results:
        with open(f"TOV_results_{model_name}.csv", "w", newline="") as f:
            write_obj = csv.writer(f)
            write_obj.writerow(["Mass", "Radius", "Pressure", "Type"])
            
            for m, r, p in model_results:
                write_obj.writerow([m, r, p, 0])
                
                                              