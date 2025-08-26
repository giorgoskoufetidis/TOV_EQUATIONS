## Plots


### Polytropic EOS $y_R$ Vs Mass (M☉)

![First group of EOS equations](/output5.png)


### Polytropic EOS $k_2$ Vs Mass (M☉)
![First group of EOS equations](/output6.png)


### Quark EOS $y_R$ Vs Mass (M☉)
![First group of EOS equations](/Machine%20Learning/yr_vs_mass.png)

### Quark EOS $y_R^{\mathrm{ext}}$ Vs Mass (M☉)
![First group of EOS equations](/Machine%20Learning/yr_ext_vs_mass.png)

### Quark EOS $k_2$ Vs Mass (M☉)
![First group of EOS equations](/Machine%20Learning/k2_vs_mass.png)

### Quark EOS $β$ Vs Mass (M☉)
![First group of EOS equations](/Machine%20Learning/b_vs_mass.png)

## Sampling from each EOS for Neutron and Quark Stars

![First group of EOS equations](/Machine%20Learning/output3.png)

![First group of EOS equations](/Machine%20Learning/output4.png)
### Quark Stars

![First group of EOS equations](/Machine%20Learning/output1.png)


### Samples from Neutron and Quark Stars 

#### Mass, Radius, $k_2$

![First group of EOS equations](/Machine%20Learning/M_R_K2.png)

#### Mass, Radius, $y_R$

![First group of EOS equations](/Machine%20Learning/M_R_Y.png)

#### Mass, Radius, $y_R^{\mathrm{ext}}$
![First group of EOS equations](/Machine%20Learning/M_R_Yext.png)


## KNN Results

Using Mass, Radius and $k_2$ 

![First group of EOS equations](/Machine%20Learning/knn_k2.png)

accuracy score: 99.59%
    

Using Mass, Radius and $y_R$

![First group of EOS equations](/Machine%20Learning/knn_y.png)

accuracy score: 99.23%
   


Using Mass, Radius, Pressure, $y_R$

![First group of EOS equations](/Machine%20Learning/knn_p.png)

accuracy score: 96.75%
          
Using Mass, Radius, $y_R$ for Neutron Star and $y_R^{\mathrm{ext}}$ for Quark Star


![First group of EOS equations](/Machine%20Learning/knn_y_ext.png)

accuracy score: 99.98%

## Random Forest Results


Using Mass, Radius, $k_2$

![First group of EOS equations](/Machine%20Learning/random_forest_k2.png)

accuracy score: 100%


Using Mass, Radius, $y_R$

![First group of EOS equations](/Machine%20Learning/random_forest_y.png)

accuracy score: 99.13%


Using Mass, Radius, Pressure, $y_R$

![First group of EOS equations](/Machine%20Learning/random_forest__pressurek_y.png)

accuracy score: 98.86%



Using Mass, Radius, $y_R$ for Neutron Star and $y_R^{\mathrm{ext}}$ for Quark Star

![First group of EOS equations](/Machine%20Learning/random_forest_y_ext.png)

accuracy score:99.99%

## Decision Tree


Using Mass, Radius, $k_2$ 

![First group of EOS equations](/Machine%20Learning/decision_trees_k2.png)

accuracy score: 100.00%

Using Mass, Radius, $y_R$

![First group of EOS equations](/Machine%20Learning/decision_trees_y.png)

accuracy score: 99.02%

Using Mass, Radius, Pressure, $y_R$

![First group of EOS equations](/Machine%20Learning/decision_trees_pressure_y.png)

accuracy score: 99.99%



Using Mass, Radius, $y_R$ for Neutron Star and $y_R^{\mathrm{ext}}$ for Quark Star

![First group of EOS equations](/Machine%20Learning/decision_trees_y_ext.png)

accuracy score: 99.97%

| Method          | Features Used                                     | Accuracy (%) |
|-----------------|---------------------------------------------------|--------------|
| **KNN**         | Mass, Radius, $k_2$                                  | 99.59        |
|                 | Mass, Radius, $y_R$                                  | 99.23        |
|                 | Mass, Radius, Pressure, $y_R$                       | 96.75        |
|                 | Mass, Radius, $y_R$ (NS), $y_R^{\mathrm{ext}}$ (QS)                | 99.98        |
| **Random Forest** | Mass, Radius, $k_2$                                | 100.00       |
|                 | Mass, Radius, $y_R$                                  | 99.13        |
|                 | Mass, Radius, Pressure, $y_R$                        | 98.86        |
|                 | Mass, Radius, $y_R$ (NS), $y_R^{\mathrm{ext}}$ (QS)                | 99.99        |
| **Decision Tree** | Mass, Radius, $k_2$                                | 100.00       |
|                 | Mass, Radius, $y_R$                                  | 99.02        |
|                 | Mass, Radius, Pressure, $y_R$                        | 99.99        |
|                 | Mass, Radius, $y_R$ (NS), $y_R^{\mathrm{ext}}$ (QS)                | 99.97        |
