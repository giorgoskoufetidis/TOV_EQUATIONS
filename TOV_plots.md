# TOV Equation and Equation of State (EOS) Models Documentation

## Tolman-Oppenheimer-Volkoff (TOV) Equation

The **TOV equation** is used to model the structure of a spherically symmetric object, such as a star, under hydrostatic equilibrium. The equation balances the gravitational force and pressure gradient:

$$
\frac{dP(r)}{dr} = -\frac{(P(r) + \rho(r)) \left( \frac{GM(r)}{r^2} + 4 \pi r \right)}{r \left( 1 - \frac{2GM(r)}{r} \right)}
$$

Where:
- \( P(r) \) is the pressure at radius \( r \),
- \( rho(r) \) is the energy density at radius \( r \),
- \( M(r) \) is the mass inside radius \( r \),
- \( G \) is the gravitational constant,
- \( r \) is the radial coordinate.

## Equation of State (EOS)

The **Equation of State (EOS)** defines the relationship between pressure and energy density. The provided models include different EOS, which can be used to describe different types of matter within a star.

### Crust Equation of State Models

#### Crust Equations

The following are the **Crust equations** based on the provided code, with the corresponding pressure ranges for each equation:

1. **Crust Equation 1** (Valid for \( 9.34375 × 10⁻⁵ ≤ P ≤ 0.184 \)):

$$
E(P) = 103.17338 \left( 1 - e^{-\frac{P}{0.38527}} \right) + 7.34979 \left( 1 - e^{-\frac{P}{0.01211}} \right) + 0.00873
$$

2. **Crust Equation 2** (Valid for \( 4.1725 × 10⁻⁸ ≤ P < 9.34375 × 10⁻⁵ \)):

$$
E(P) = 0.00015 + 0.00203 \left( 1 - e^{-\frac{P}{344827.5}} \right) + 0.10851 \left( 1 - e^{-\frac{P}{7692.3076}} \right)
$$

3. **Crust Equation 3** (Valid for \( 1.44875 × 10⁻¹¹ ≤ P < 4.1725 × 10⁻⁸ \)):

$$
E(P) = 0.0000051 \left( 1 - e^{-\frac{P}{0.2373 \times 10^{10}}} \right) + 0.00014 \left( 1 - e^{-\frac{P}{0.4020 \times 10^8}} \right)
$$

4. **Crust Equation 4** (Valid for \( P < 1.44875 × 10⁻¹¹ \)):

$$
E(P) = 10^{31.93753 + 10.82611 \cdot \log_{10}(P) + 1.29312 \cdot \left( \log_{10}(P) \right)^2 + 0.08014 \cdot \left( \log_{10}(P) \right)^3 + 0.00242 \cdot \left( \log_{10}(P) \right)^4 + 0.000028 \cdot \left( \log_{10}(P) \right)^5}
$$

### EOS Models

The following are the **EOS models** from the provided code:

1. **MDI_1**:

$$
E(P) = 4.1844 \cdot P^{0.81449} + 95.00135 \cdot P^{0.31736}
$$

2. **MDI_2**:

$$
E(P) = 5.97365 \cdot P^{0.77374} + 89.24 \cdot P^{0.30993}
$$

3. **MDI_3**:

$$
E(P) = 15.55 \cdot P^{0.666} + 76.71 \cdot P^{0.247}
$$

4. **MDI_4**:

$$
E(P) = 25.99587 \cdot P^{0.61209} + 65.62193 \cdot P^{0.15512}
$$

5. **NLD**:

$$
E(P) = 119.05736 + 304.80445 \cdot \left( 1 - e^{-\frac{P}{48.61465}} \right) + 33722.34448 \cdot \left( 1 - e^{-\frac{P}{17499.47411}} \right)
$$

6. **HHJ_1**:

$$
E(P) = 1.78429 \cdot P^{0.93761} + 106.93652 \cdot P^{0.31715}
$$

7. **HHJ_2**:

$$
E(P) = 1.18961 \cdot P^{0.96539} + 108.40302 \cdot P^{0.31264}
$$

8. **SKa**:

$$
E(P) = 0.53928 \cdot P^{1.01394} + 94.31452 \cdot P^{0.35135}
$$

9. **SkI4**:

$$
E(P) = 4.75668 \cdot P^{0.76537} + 105.722 \cdot P^{0.2745}
$$

10. **HLPS_2**:

$$
E(P) = 172.858 \cdot \left( 1 - e^{-\frac{P}{22.8644}} \right) + 2777.75 \cdot \left( 1 - e^{-\frac{P}{1909.97}} \right) + 161.553
$$

11. **HLPS_3**:

$$
E(P) = 131.811 \cdot \left( 1 - e^{-\frac{P}{4.41577}} \right) + 924.143 \cdot \left( 1 - e^{-\frac{P}{523.736}} \right) + 81.5682
$$

12. **SCVBB**:

$$
E(P) = 0.371414 \cdot P^{1.08004} + 109.258 \cdot P^{0.351019}
$$

13. **WFF_1**:

$$
E(P) = 0.00127717 \cdot P^{1.69617} + 135.233 \cdot P^{0.331471}
$$

14. **WFF_2**:

$$
E(P) = 0.00244523 \cdot P^{1.62962} + 122.076 \cdot P^{0.340401}
$$

15. **PS**:

$$
E(P) = 9805.95 \cdot \left( 1 - e^{-0.000193624 \cdot P} \right) + 212.072 \cdot \left( 1 - e^{-0.401508 \cdot P} \right) + 1.69483
$$

16. **W**:

$$
E(P) = 0.261822 \cdot P^{1.16851} + 92.4893 \cdot P^{0.307728}
$$

17. **BGP**:

$$
E(P) = 0.0112475 \cdot P^{1.59689} + 102.302 \cdot P^{0.335526}
$$

18. **BL_1**:

$$
E(P) = 0.488686 \cdot P^{1.01457} + 102.26 \cdot P^{0.355095}
$$

19. **BL_2**:

$$
E(P) = 1.34241 \cdot P^{0.910079} + 100.756 \cdot P^{0.354129}
$$

20. **DH**:

$$
E(P) = 39.5021 \cdot P^{0.541485} + 96.0528 \cdot P^{0.00401285}
$$

21. **APR_1**:

$$
E(P) = 0.000719964 \cdot P^{1.85898} + 108.975 \cdot P^{0.340074}
$$

# Quark Star Equation of State (EOS) Generator


- Computes the stability function for a given quark system.
- Determines the stability limit for the bag constant (`B`) based on physical constraints.
- Implements a model that calculates the energy density as a function of pressure.

## Formulas Used

### Alpha Parameter
```math
\alpha = -\frac{m_s^2}{6} + \frac{2\delta^2}{3}
```

### Energy Density
```math
\epsilon = 3P + 4B - \frac{9 \alpha m^2}{\pi^2 \hbar c^3}
```
where
```math
m = \sqrt{-3\alpha + \sqrt{\frac{4}{3} \pi^2 (B + P) \hbar c^3 + 9\alpha^2}}
```

### Stability Function
```math
\beta = -\frac{m_s^2 m_n^2}{12 \pi^2} + \frac{\delta^2 m_n^2}{3 \pi^2} + \frac{m_n^4}{108 \pi^2}
```
```math
\beta_{stability} = \frac{\beta}{(\hbar c)^3}
```

### Stability Point (Upper Limit for B)
```math
B_{max} = \frac{-\frac{m_s^2 m_n^2}{12 \pi^2} + \frac{\delta^2 m_n^2}{3 \pi^2} + \frac{m_n^4}{108 \pi^2}}{(\hbar c)^3}
```



## Plots

### First group of EOS equations
![First group of EOS equations](/Figure_1.png)

### Second group of EOS equations
![Second group of EOS equations](/Figure_2.png)


![Quark stars](/Figure_3.png)
