import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

FIGURES_DIR = "../report/figures/";

np.random.seed(1140);

dataset = np.loadtxt("../gmsl_2020rel1_seasons_retained.txt");
N = dataset.shape[0];
idx = np.arange(N);
np.random.shuffle(idx);

train_idx = idx[:int(N*0.8)];
val_idx   = idx[int(N*0.8):];
x_train, y_train = dataset[train_idx, 0], dataset[train_idx, 1];
x_val,   y_val   = dataset[  val_idx, 0], dataset[  val_idx, 1];
x_train = x_train.reshape(-1,1);
y_train = y_train.reshape(-1,1);
x_val = x_val.reshape(-1,1);
y_val = y_val.reshape(-1,1);

plt.figure()
plt.scatter( x_train, y_train, color='blue', label='Training',   s=5 );
plt.scatter( x_val,   y_val,   color='red',  label='Validation', s=5 );
plt.title("Global Mean Sea Level (Seasonal Signals Retained)");
plt.xlabel("Year");
plt.ylabel("Mean Sea Level [mm]");
plt.legend();
plt.grid(True);
plt.savefig(f"{FIGURES_DIR}/dataset.png");

# Normalize dataset base on training data
ss_x = StandardScaler();
ss_x.fit( x_train );
x_train = ss_x.transform(x_train)
x_val   = ss_x.transform(x_val  )

ss_y = StandardScaler();
ss_y.fit( y_train );
y_train = ss_y.transform(y_train)
y_val   = ss_y.transform(y_val  )



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part A
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

lr = LinearRegression();
lr.fit(x_train, y_train);
b_lr , w_lr = lr.intercept_[0], lr.coef_[0][0]
y_pred = b_lr + x_val*w_lr;
mse = (( y_pred-y_val)**2).mean()
print(f"[LR] b: {b_lr:.4} w: {w_lr:.4} msg: {mse:.4}");

x_lr = np.linspace( min(x_train), max(x_train), 10 );
y_lr = b_lr + (x_lr*w_lr);

plt.figure()
#plt.scatter( x_train, y_train, color='blue', label='Training',   s=5 );
plt.scatter( x_val, y_val, color='red', label='Validation',   s=5 );
plt.plot( x_lr, y_lr, color='black', label='Linear Regression', linewidth=3 );
plt.title("Golden Prediction");
plt.grid(True);
plt.legend();
plt.savefig(f"{FIGURES_DIR}/linear_regression.png");

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part B
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

b_depth = 3;
w_depth = 3;
b_range = np.linspace( b_lr - b_depth, b_lr + b_depth, 100 );
w_range = np.linspace( w_lr - w_depth, b_lr + w_depth, 100 );
b_surf, w_surf = np.meshgrid( b_range, w_range );
y_surf = np.apply_along_axis( func1d=lambda x: b_surf + w_surf*x,
                              axis=1,
                              arr=x_train );

all_labels = y_train.reshape(-1,1,1);
all_errors = (y_surf-all_labels);
loss_surf = (all_errors**2).mean(axis=0);

fig = plt.figure();
ax = fig.add_subplot( 111, projection='3d')
ax.plot_surface( b_surf, w_surf, loss_surf, rstride=1, cstride=1, alpha=.5, cmap=plt.cm.jet, linewidth=0, antialiased=True);
ax.scatter(b_lr, w_lr, color='red', s=20);
ax.set_xlabel('b');
ax.set_ylabel('w');
ax.set_title('Loss Surface');
ax.view_init(40, 260)
plt.savefig(f"{FIGURES_DIR}/loss_surface.png");

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part C
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
iterations = [];
best_mse = float('inf');
N_ITR = 1000;
for itr in range(N_ITR):
    b_rand = np.random.randn();
    w_rand = np.random.randn();
    y_pred = b_rand + x_val*w_rand;
    mse = (( y_pred-y_val)**2).mean()
    if( mse < best_mse ):
        iterations.append([b_rand, w_rand, float(mse)]);
        best_mse = mse;

iterations = np.array(iterations);
print(f"[Naive] b: {iterations[-1][0]:.4}, w: {iterations[-1][1]:.4}, mse: {iterations[-1][2]:.4}");

fig = plt.figure();
ax = fig.add_subplot( 111, projection='3d')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part D
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
ax.plot_surface( b_surf, w_surf, loss_surf, rstride=1, cstride=1, alpha=.5, cmap=plt.cm.jet, linewidth=0, antialiased=True);
ax.scatter(b_lr, w_lr, color='red', s=20);
ax.scatter(iterations[0][0], iterations[0][1], color='blue', s=40);
ax.scatter(iterations[1:-1:,0], iterations[1:-1:,1,], color='black', s=20);
ax.scatter(iterations[-1][0], iterations[-1][1], color='blue', s=40);
ax.set_xlabel('b');
ax.set_ylabel('w');
ax.set_title('Loss Surface');
ax.view_init(40, 260)
plt.savefig(f"{FIGURES_DIR}/loss_surface_naive.png");

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part E
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

for N_ITR in [ 1000, 2000, 5000, 10000 ]:
    iterations = [];
    best_mse = float('inf');
    for itr in range(N_ITR):
        b_rand = np.random.randn();
        w_rand = np.random.randn();
        y_pred = b_rand + x_val*w_rand;
        mse = (( y_pred-y_val)**2).mean()
        if( mse < best_mse ):
            iterations.append([b_rand, w_rand, float(mse)]);
            best_mse = mse;

    iterations = np.array(iterations);
    print(f"[Naive-{N_ITR}] b: {iterations[-1][0]:.4}, w: {iterations[-1][1]:.4}, mse: {iterations[-1][2]:.4}\n\tHits: {iterations.shape[0]} Missed: {N_ITR-iterations.shape[0]}");


plt.show();
