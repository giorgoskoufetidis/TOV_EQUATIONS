
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:13:20 2024

@author: User
"""


import pandas as pd
import matplotlib.pyplot as plt


file_path = "TOV_results_for_multiple_models (1).txt"

results = {}

with open(file_path, "r") as f:
    lines = f.readlines()

current_model = None
data = []

for line in lines:
    line = line.strip()
    if line.startswith("# model:"):
            current_model = line.split(":")[1].strip()
            df = pd.DataFrame(data,columns = ["Mass","Radius","Pressure"])
            results[current_model] = df
            data = []
    elif line and not line.startswith("Mass") and not line.startswith("#"):
        values = list(map(float,line.split()))
        data.append(values)
if current_model and data:
    df = pd.DataFrame(data,columns = ["Mass","Radius","Pressure"])
    results[current_model] = df


#for model_name, df in results.items():
#    print(f"Model:{model_name}")

model_names = list(results.keys())
split_index = len(model_names) // 2
group_1 = model_names[:split_index]
group_2 = model_names[split_index:]

# Plot 1: First half of the models
plt.figure(figsize=(10, 6))
for model_name in group_1:
    df = results[model_name]
    plt.plot(df["Radius"], df["Mass"], label=f"{model_name}: Mass vs Radius")
plt.xlabel("Radius (km)")
plt.ylabel("Mass (M☉)")
plt.title("Mass-Radius Relation ")
plt.legend()
plt.grid(True)

plt.savefig("Figure_1.jpg")

plt.figure(figsize=(10, 6))
for model_name in group_2:
    df = results[model_name]
    plt.plot(df["Radius"], df["Mass"], label=f"{model_name}: Mass vs Radius")
    plt.xlim(8,30)
plt.xlabel("Radius (km)")
plt.ylabel("Mass (M☉)")
plt.title("Mass-Radius Relation ")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Figure_2.jpg")


'''
# Example: Plot Pressure vs Radius for all models
for model_name, df in results.items():
    plt.figure(figsize=(8, 6))
    plt.plot(df["Radius"], df["Pressure"], label=f"{model_name}: Pressure vs Radius", color="green")
    plt.xlabel("Radius (km)")
    plt.ylabel("Pressure")
    plt.title(f"Pressure-Radius Relation ({model_name})")
    plt.legend()
    plt.grid(True)
  '''

