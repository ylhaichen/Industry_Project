import numpy as np, pandas as pd, os, matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from ahrs.filters import Mahony, Madgwick
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

# --- 基础读取同前 -----------------------------------------------------------
SID  = 'S_36833'
CSV  = os.path.join('re-synced with all sensors/synced_data',
                    SID, 'PANORAMIC_RIGHT_all_sensors.csv')
df = (pd.read_csv(CSV, parse_dates=['timestamp'])
        .dropna(subset=['GYRO_x','GYRO_y','GYRO_z',
                        'HE_ACC_x','HE_ACC_y','HE_ACC_z'])
        .sort_values('timestamp').reset_index(drop=True))

df['time'] = (df['timestamp']-df['timestamp'].iloc[0]).dt.total_seconds()
time = df['time'].to_numpy()
gyro = df[['GYRO_x','GYRO_y','GYRO_z']].to_numpy()*np.pi/180
acc  = df[['HE_ACC_x','HE_ACC_y','HE_ACC_z']].to_numpy()
fs   = 52.0

clip_info = {
    'Lower_limb': ('05:34:31 PM', '05:34:41 PM'),
    'Head'      : ('05:35:54 PM', '05:36:06 PM'),
    'Upper_limb': ('05:37:17 PM', '05:37:35 PM'),
    'Chest'     : ('05:41:26 PM', '05:41:46 PM'),
}

# --- Mahony 去重力 ----------------------------------------------------------
ma = Mahony(frequency=fs)
q  = np.array([1.,0.,0.,0.])
g_ma = np.zeros_like(acc)
for k in range(len(time)):
    q = ma.updateIMU(q, gyr=gyro[k], acc=acc[k]); q /= np.linalg.norm(q)
    g_ma[k] = R.from_quat(q).apply([0,0,9.81])        # ← 无 inv()
lin_ma = acc - g_ma

# --- Madgwick 去重力 --------------------------------------------------------
md = Madgwick(sampleperiod=1/fs, beta=0.1)
q  = np.array([1.,0.,0.,0.])
g_md = np.zeros_like(acc)
for k in range(len(time)):
    q = md.updateIMU(q, gyr=gyro[k], acc=acc[k]); q /= np.linalg.norm(q)
    g_md[k] = R.from_quat(q).apply([0,0,9.81])
lin_md = acc - g_md

# --- Upper-limb 区段时间 -----------------------------------------------------
base_date = df['timestamp'].iloc[0].date()
to_sec = lambda s: (pd.to_datetime(f'{base_date} {s}') -
                    df['timestamp'].iloc[0]).total_seconds()
t0, t1 = map(to_sec, ('05:37:17 PM', '05:37:35 PM'))   # upper-limb

mask  = (time>=t0)&(time<=t1)
t_seg = time[mask] - time[mask][0]                      # 相对时间
a_ma  = lin_ma[mask]
a_md  = lin_md[mask]

# --- 双积分 → 位置 ----------------------------------------------------------
# 速度: v(t) = ∫ a dt ； 位置: p(t) = ∫ v dt
v_ma = cumulative_trapezoid(a_ma, t_seg, initial=0, axis=0)
p_ma = cumulative_trapezoid(v_ma, t_seg, initial=0, axis=0)

v_md = cumulative_trapezoid(a_md, t_seg, initial=0, axis=0)
p_md = cumulative_trapezoid(v_md, t_seg, initial=0, axis=0)

# --- 二阶差分反推加速度 ------------------------------------------------------
# 先差分速度 (一次微分) 再差分得到加速度
dt  = np.diff(t_seg)[:,None]              # (N-1,1)
a_re_ma = np.vstack(([0,0,0],
                     np.diff(v_ma, axis=0)/dt))
a_re_md = np.vstack(([0,0,0],
                     np.diff(v_md, axis=0)/dt))

# ── 绘图 1: 3-D 轨迹对比 -----------------------------------------------------
fig = plt.figure(figsize=(7,5)); ax = fig.add_subplot(111, projection='3d')
ax.plot(*p_ma.T, label='Mahony')
ax.plot(*p_md.T, label='Madgwick', ls='--')
ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
       title='Upper-limb 3-D trajectory (double-integration)')
ax.legend(); plt.tight_layout(); plt.show()

# ── 绘图 2: 原始加速度 + 去重力 + 重建加速度 -------------------------------
labels  = ['X', 'Y', 'Z']
colors  = ['C0', 'C1', 'C2']

fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

for i, lab in enumerate(labels):
    # --- raw accel (with gravity) ----------------------------------------
    axs[i].plot(t_seg, a_ma[:, i] + g_ma[mask, i], color='k',
                alpha=0.35, label='Raw acc (with g)')

    # --- Mahony linear and recon ----------------------------------------
    axs[i].plot(t_seg, a_ma[:, i],           c=colors[i],   lw=1.4,
                label='Mahony a_'+lab)
    axs[i].plot(t_seg, a_re_ma[:, i],        c=colors[i],   ls=':',
                label='Recon Mahony')

    # --- Madgwick linear and recon --------------------------------------
    axs[i].plot(t_seg, a_md[:, i],           c=colors[i],   lw=1.4, ls='--',
                label='Madgwick a_'+lab)
    axs[i].plot(t_seg, a_re_md[:, i],        c=colors[i],   ls='-.',
                label='Recon Madgwick')

    axs[i].set_ylabel(f'{lab} (m/s²)')
    axs[i].grid(ls=':')

axs[-1].set_xlabel('Time in segment (s)')
axs[0].set_title('Upper-limb:Gravity-removed vs reconstructed acceleration')
axs[0].legend(ncol=2, fontsize=8)
plt.tight_layout(); plt.show()
