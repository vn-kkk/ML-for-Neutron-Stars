# 2\_PP\_EOS\_emulator





*This is the piecewise polytropic solution*





* **PP\_TOV\_Emulator.ipynb:** Bhaskar's notebook to predict m-r relations for NS using PP EOS. But had small errors in ML pipeline. Still worked.

 	\* **EOS\_dataset.npy:** The dataset containing 10000 samples of \[logroh\_c, gamma2 and gamma3]. gamma1 and logp were fixed.





* **EMULATOR\_2:** Same as above





* **EMULATOR\_3:** Extended from 2 to take in gamma1 as an input parameter as well

 	\* The dataset containing 10000 samples of \[logroh\_c, gamma1, gamma2 and gamma3]. logp was fixed







* **EMULATOR\_4:** Extended from 2 to take in gamma1 and logp as input parameters

 	\* The dataset containing 10000 samples of \[logroh\_c, logp, gamma1, gamma2 and gamma3]

