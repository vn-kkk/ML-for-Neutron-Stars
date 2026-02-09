# Naming convention:

Eg: 3\_Emulator\_PIRNN\_Hreal(EOS+M)\_R

&nbsp;   Order of folder created\_Emulator\_Kind of NN\_kind of EOS and data(input parameters)\_Output parameters

# -----------------------------------------





### 0\_Emulator\_NN\_PEOS(EOS+CD)\_MR

Simple NN trained on Polytropic EOS parameters and Central density to predict Mass and Radius: Simple\_TOV\_Emulator.ipynb

From Bhaskar.



IDEAL EXAMPLE







### 1\_Emulator\_NN\_NuclearEOS(EOS+CP)\_MR

NN trained on Nuclear EOS parameters and central pressure to predict Mass, Radius and Tidal Deformability: Kay\_TOV\_Emulator.ipynb and TOV\_Emulator\_Kay

Contains Different Emulators to predict individual parameters only: Kay\_TOV\_...\_Emulator.ipynb

&nbsp;	Mass

&nbsp;	MaxMass

&nbsp;	Radius	

&nbsp;	TD



Also contains a lecture notebook: Kay\_ML\_for\_physicists.ipynb



NOT GOOD ENOUGH







### 2\_Emulator\_NN\_PPEOS(EOS+CD)\_MR

NN trained on Piecewise Polytropic EOS and central density to predict Mass and Radius:PP\_TOV\_Emulator.ipynb

Different versions of the Emulator exists within the folder. 

&nbsp;	Emulator\_2

&nbsp;	Emulator\_3

&nbsp;	Emulator\_4



NOT GOOD ENOUGH







### 3\_Emulator\_PIRNN\_Hreal(EOS+M)\_R

Final version of the code before Christmas. But the dataset was created wrongly!

Emulator\_real.ipynb is the working code for small local runs. The big folders have the cluster code within them for full runs.

#### 1000files ✅

CORRECT.

#### 399700files\_R ✨✅

CORRECT.







### 4\_Emulator\_PIRNN\_Hreal(EOS+M)\_RTD

#### 1000files ✅

CORRECT.







### 5\_Emulator\_PIRNN\_Hreal(EOS+interpM)\_R

#### 1000files ✅

CORRECT.

#### 399700files\_interpM\_R ✨✅

CORRECT







### 6\_Emulator\_PIRNN\_Hreal(EOS+CP)\_MR

#### 1000files ✅

CORRECT.

#### 399700files\_MR ❌

Delete if not used







### 7\_Emulator\_PIRNN\_Hreal(EOS+CP)\_MRTD

#### 1000files ✅

CORRECT.

399700files\_MRTD ✨ᯓ🏃🏻‍♀️‍➡️

Cluster running ...







### 8\_Emulator\_PIRNN\_PPEOS(EOS+CD)\_MR

Adapting the PIRNN to train on PPEOS

