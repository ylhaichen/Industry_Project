import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from ahrs.filters import Mahony
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

# 1) Load and preprocess IMU data -------------------------------------------
SID = 'S_36833'
CSV = os.path.join('re-synced with all sensors/synced_data', SID,
                   'PANORAMIC_RIGHT_all_sensors.csv')

df = (pd.read_csv(CSV, parse_dates=['timestamp'])
        .dropna(subset=['GYRO_x','GYRO_y','GYRO_z',
                        'HE_ACC_x','HE_ACC_y','HE_ACC_z'])
        .sort_values('timestamp')
        .reset_index(drop=True))
df['time'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

time = df['time'].to_numpy()
gyro = df[['GYRO_x','GYRO_y','GYRO_z']].to_numpy() * np.pi/180
acc  = df[['HE_ACC_x','HE_ACC_y','HE_ACC_z']].to_numpy()

fs = 52.0  # sampling rate  
dt = 1.0 / fs

# 2) Define scratching intervals (human‐readable) --------------------------
clip_info = {
    'Lower_limb': ('05:34:31 PM', '05:34:36 PM'),
    'Head'      : ('05:35:54 PM', '05:35:59 PM'),
    'Upper_limb': ('05:37:17 PM', '05:37:22 PM'),
    'Chest'     : ('05:41:26 PM', '05:41:31 PM'),
}

# Convert to seconds relative to df start -----------------------------------
base_date = df['timestamp'].iloc[0].date()
to_sec = lambda s: (pd.to_datetime(f'{base_date} {s}')
                    - df['timestamp'].iloc[0]).total_seconds()

for tag, (s0, s1) in clip_info.items():
    clip_info[tag] = (to_sec(s0), to_sec(s1))

# 3) Mahony filter for gravity removal --------------------------------------
mahony = Mahony(frequency=fs)
q = np.array([1.,0.,0.,0.])
g_est = np.zeros_like(acc)

for k in range(len(time)):
    q = mahony.updateIMU(q, gyr=gyro[k], acc=acc[k])
    q /= np.linalg.norm(q)
    # rotate world gravity [0,0,9.81] into sensor frame
    g_est[k] = R.from_quat(q).apply([0,0,9.81])

lin_ma = acc - g_est    # linear acceleration, gravity removed
# design a 0.2 Hz high-pass filter
b, a = butter(2, 0.2/(fs/2), btype='high')
lin_ma_hp = filtfilt(b, a, lin_ma, axis=0)
# 4) Compute and plot velocity for each area -------------------------------
areas  = ['Lower_limb', 'Head', 'Upper_limb', 'Chest']
colors = ['C0','C1','C2']
labels = ['X','Y','Z']

fig, axs = plt.subplots(len(areas), 1, figsize=(10, 8), sharex=False)

for idx, area in enumerate(areas):
    t0, t1 = clip_info[area]
    mask   = (time >= t0) & (time <= t1)
    t_rel  = time[mask] - time[mask][0]

    a_seg = lin_ma_hp[mask]  # shape (N,3)
    # integrate once to get velocity
    vel = np.cumsum(a_seg, axis=0) * dt  
    
    # plot X,Y,Z velocity
    for i in range(3):
        axs[idx].plot(t_rel, vel[:,i],
                      color=colors[i],
                      label=labels[i] if idx==0 else "")
    axs[idx].set_ylabel(f'{area.replace("_"," ")}\nVelocity (m/s)')
    axs[idx].grid(ls=':')

axs[-1].set_xlabel('Time in segment (s)')
handles, lbls = axs[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc='upper right', ncol=3)
plt.suptitle('Estimated Velocity (Mahony) for Four Scratching Areas')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

plt.figure(figsize=(10,5))

for area in areas:
    # extract time segment
    t0, t1 = clip_info[area]
    mask   = (time >= t0) & (time <= t1)
    t_rel  = time[mask] - time[mask][0]

    # get HP‐filtered accel and integrate
    a_seg = lin_ma_hp[mask]           # (N,3)
    vel   = np.cumsum(a_seg, axis=0) * dt

    # compute net speed
    speed = np.linalg.norm(vel, axis=1)

    # plot
    plt.plot(t_rel, speed, label=area.replace('_',' '))

plt.xlabel('Time in segment (s)')
plt.ylabel('Speed (m/s)')
plt.title('Net Velocity for Four Scratching Areas')
plt.legend()
plt.grid(ls=':')
plt.tight_layout()
plt.show()