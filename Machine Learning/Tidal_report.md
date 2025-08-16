## Sampling from each EOS for Neutron and Quark Stars

![First group of EOS equations](/Machine%20Learning/output3.png)

![First group of EOS equations](/Machine%20Learning/output4.png)
### Quark Stars

![First group of EOS equations](/Machine%20Learning/output1.png)


### Samples from Neutron and Quark Stars 

#### M-R-K2

![First group of EOS equations](/Machine%20Learning/M_R_K2.png)

#### M-R-yR

![First group of EOS equations](/Machine%20Learning/M_R_Y.png)

#### M-R-yR_ext
![First group of EOS equations](/Machine%20Learning/M_R_Yext.png)


## KNN Results

Using Mass, Radius and K2 

![First group of EOS equations](/Machine%20Learning/knn_k2.png)

accuracy score: 99.59%
    

Using Mass, Radius and yR

![First group of EOS equations](/Machine%20Learning/knn_y.png)

accuracy score: 99.23%
   


Using Mass, Radius, Pressure, yR

![First group of EOS equations](/Machine%20Learning/knn_p.png)

accuracy score: 96.75%
          
Using Mass, Radius, yR for Neutron Star and yR_ext for Quark Star


![First group of EOS equations](/Machine%20Learning/knn_y_ext.png)

accuracy score: 99.98%

## Random Forest Results


Using Mass, Radius, k2

![First group of EOS equations](/Machine%20Learning/random_forest_k2.png)

accuracy score: 100%


Using Mass, Radius, yR

![First group of EOS equations](/Machine%20Learning/random_forest_y.png)

accuracy score: 99.13%


Using Mass, Radius, Pressure, yR

![First group of EOS equations](/Machine%20Learning/random_forest__pressurek_y.png)
accuracy score: 98.86%



Using Mass, Radius, yR for Neutron Star and yR_ext for Quark Star

![First group of EOS equations](/Machine%20Learning/random_forest_y_ext.png)

accuracy score:99.99%

## Decision Tree


Using Mass, Radius, k2 

![First group of EOS equations](/Machine%20Learning/decision_trees_k2.png)

accuracy score: 100.00%

Using Mass, Radius, yR 

![First group of EOS equations](/Machine%20Learning/decision_trees_y.png)

accuracy score: 99.02%

Using Mass, Radius, Pressure, yR

![First group of EOS equations](/Machine%20Learning/decision_trees_pressure_y.png)

accuracy score: 99.99%



Using Mass, Radius, yR for Neutron Star and yR_ext for Quark Star

![First group of EOS equations](/Machine%20Learning/decision_trees_y_ext.png)

accuracy score: 99.97%

| Method          | Features Used                                     | Accuracy (%) |
|-----------------|---------------------------------------------------|--------------|
| **KNN**         | Mass, Radius, k2                                  | 99.59        |
|                 | Mass, Radius, yR                                  | 99.23        |
|                 | Mass, Radius, Pressure, yR                        | 96.75        |
|                 | Mass, Radius, yR (NS), yR_ext (QS)                | 99.98        |
| **Random Forest** | Mass, Radius, k2                                | 100.00       |
|                 | Mass, Radius, yR                                  | 99.13        |
|                 | Mass, Radius, Pressure, yR                        | 98.86        |
|                 | Mass, Radius, yR (NS), yR_ext (QS)                | 99.99        |
| **Decision Tree** | Mass, Radius, k2                                | 100.00       |
|                 | Mass, Radius, yR                                  | 99.02        |
|                 | Mass, Radius, Pressure, yR                        | 99.99        |
|                 | Mass, Radius, yR (NS), yR_ext (QS)                | 99.97        |
