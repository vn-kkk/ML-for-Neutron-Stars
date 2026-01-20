# Characteristics of the dataset I am using







**FILE NAMES**

**----------**

* The names of the files contains the EOS parameters:

 	Hadronic EOS parameters

 	\* Effective nuclear mass (m/100)

 	\* Slope parameter (L)

 	\* Symmetry Energy (J)

 	\* Temperature (T = 0)

 	\* Saturation density (n/1000)

 

 	Quark EOS parameters

 	\* Vector coupling (n\_v)

 	\* Dipole couplinf (n\_d)

 	\* Bag constant (B\_n or B\_p /1000)





**400,000 FILES**

**-------------**

* *Each file is a different EOS.*
* Each EOS produces multiple stable stars with different mass, radius, ...





**EACH FILE**

**---------**

* Each files contains 400 rows of these 6 columns:

 	\* **Central pressure (p\_c)**

 	\* **Mass (M)**

 	\* **Radius (R)**

 	\* **Tidal Deformability (Λ)**

 	\* Radius of quark core (R\\\_c) (0 → hadronic, >0 → hybrid)

 	\* **Baryonic mass (ρ)**



* *Each row withing a .dat file represents a different stable Neutron Star*.
* The row values are attained by solving the same EOS multiple times for different central conditions (central pressure or central mass density).

 

* *Each file produces one M-R curve.* The points that make up these curves are the 400 values within each file.
* However we cutoff the branch of the curve that exists after the maximum mass(to the left of M\_max) because those solutions are physically unstable because the star collapses as Gravitational pull exceeds the outward degeneracy pressure.





**NN Prediction**

---

* Input parameters: m/100, L, J, n/1000, n\_v, n\_d, B/1000 and M
* Output Parameters: R
