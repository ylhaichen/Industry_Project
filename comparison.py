import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from ahrs.filters import Mahony
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401

# ── 1) Load IMU ------------------------------------------------------------
FNAME = 're-synced with all sensors/synced_data/S_36833/PANORAMIC_RIGHT_all_sensors.csv'
keep  = ['GYRO_x','GYRO_y','GYRO_z','HE_ACC_x','HE_ACC_y','HE_ACC_z']
df    = (pd.read_csv(FNAME, parse_dates=['timestamp'])
           .sort_values('timestamp')
           .dropna(subset=keep)
           .reset_index(drop=True))
df[keep] = df[keep].astype(float)
df['time'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

time = df['time'].to_numpy()
gyro = df[['GYRO_x','GYRO_y','GYRO_z']].to_numpy()*np.pi/180.0   # rad/s
acc  = df[['HE_ACC_x','HE_ACC_y','HE_ACC_z']].to_numpy()         # m/s²
fs   = 52.0                              # Hz

# --------------------------------------------------------------------------
# 2-A)  Method-1  ——  orientation-based gravity removal
# --------------------------------------------------------------------------
mahony = Mahony(frequency=fs, Kp=1.0, Ki=0.0)
q      = np.array([1.,0.,0.,0.])          # assume start aligned with world
g_est_m1 = np.zeros_like(acc)

for k in range(len(time)):
    q = mahony.updateIMU(q, gyr=gyro[k], acc=acc[k])
    q /= np.linalg.norm(q)                # safeguard
    # gravity in sensor frame:
    g_est_m1[k] = R.from_quat(q).inv().apply([0,0,9.81])

lin_acc_m1 = acc - g_est_m1

# --------------------------------------------------------------------------
# 2-B)  Method-2  ——  filtfilt Butterworth LP
# --------------------------------------------------------------------------
cutoff, order = 0.3, 2
b, a   = butter(order, cutoff/(fs/2), btype='low')
g_est_m2 = filtfilt(b, a, acc, axis=0)
lin_acc_m2 = acc - g_est_m2

# --------------------------------------------------------------------------
# 3)  Integrate each linear-acc set with ZUPT
# --------------------------------------------------------------------------
def integrate(lin_acc):
    vel = np.zeros_like(lin_acc)
    pos = np.zeros_like(lin_acc)
    for i in range(1, len(time)):
        dt = time[i]-time[i-1]
        vel[i] = vel[i-1] + lin_acc[i]*dt
        if np.linalg.norm(lin_acc[i]) < 0.05:   # ZUPT
            vel[i] = 0.0
        pos[i] = pos[i-1] + vel[i]*dt
    return vel, pos

vel1, pos1 = integrate(lin_acc_m1)
vel2, pos2 = integrate(lin_acc_m2)

# --------------------------------------------------------------------------
# 4)  PLOTS
# --------------------------------------------------------------------------
labels = ['X','Y','Z']; colours = ['C0','C1','C2']

# 4-a  3-D trajectories
fig = plt.figure(figsize=(9,6)); ax=fig.add_subplot(111, projection='3d')
ax.plot(*pos1.T, label='Method-1 (gyro+Mahony)',  lw=1.2)
ax.plot(*pos2.T, label='Method-2 (LP filtfilt)',  lw=1.2)
ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
       title='3-D Trajectory comparison'); ax.legend(); plt.tight_layout(); plt.show()

# 4-b  Linear-acc components
fig,axs = plt.subplots(3,1, figsize=(10,7), sharex=True)
for i,lab in enumerate(labels):
    axs[i].plot(time, lin_acc_m1[:,i], c=colours[i], label='a_lin M1')
    axs[i].plot(time, lin_acc_m2[:,i], c=colours[i], ls='--', label='a_lin M2')
    axs[i].set_ylabel(f'{lab}  (m/s²)'); axs[i].grid(ls=':')
axs[-1].set_xlabel('Time (s)')
axs[0].set_title('Linear acceleration components')
axs[0].legend(ncol=2); plt.tight_layout(); plt.show()

# 4-c  Velocity drift comparison (optional)
plt.figure(figsize=(9,4))
plt.plot(time, np.linalg.norm(vel1,axis=1), label='‖vel‖ M1')
plt.plot(time, np.linalg.norm(vel2,axis=1), '--', label='‖vel‖ M2')
plt.xlabel('Time (s)'); plt.ylabel('m/s'); plt.title('Velocity magnitude')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
