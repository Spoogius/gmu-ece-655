# Introduce measurement error - Using project example
v_r = dataset[:,2]; # Voltage at current limiting resitor
v_s = dataset[:,1]; # DC Source Voltage

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