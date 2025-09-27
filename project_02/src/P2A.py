import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.preprocessing import StandardScaler;
import os;
# ----------------------------------
#          Part A
# ----------------------------------

np.random.seed(1140);

# Load full LT Spice simulated dataset ~2000 points -> skip 775 points until diode is active
dataset = np.loadtxt("../lt_spice/project_2_lt_spice.txt", skiprows=775);


N = 50;
idx = np.arange(dataset.shape[0]);
np.random.shuffle( idx );

# Trim dataset to N points
dataset = dataset[idx[0:N:], :];

# Introduce measurement error - Using project example
v_r = dataset[:,2];
v_s = dataset[:,1];

# Clip to 4 decimal palces
v_r = (np.round(v_r*100000))/100000;

# Create noiseM - Normal distribtuion clipped between +- 0.05% = 0.0005
noiseM_scale = np.random.normal( loc=0, scale=0.0005, size=v_r.shape);
# Create noiseQ - unitform distribution -0.0002 -> + 0.0002
noiseQ = np.random.randint(-2, 2, size=v_r.shape)/10000;

# Add noise to voltage measurement
v_r = (v_r*(1+noiseM_scale)) + (noiseQ);

# Compute Diode current
R = 15; # Resistor Value
i_d = v_r/R;

# Compute diode voltage
v_d = v_s - v_r;


plt.figure();
plt.scatter(v_d,1000* i_d );
#plt.scatter(v_d, np.log(i_d) );
plt.xlabel("V_d (V)");
plt.ylabel("I_d (mA)");

dataset = np.empty((N,3));
dataset[:,0] = v_d;
dataset[:,1] = i_d;
dataset[:,2] = np.log(i_d);