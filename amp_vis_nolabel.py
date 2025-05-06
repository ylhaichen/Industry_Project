import pandas as pd
import matplotlib.pyplot as plt

# --- Load IMU data ---
imu = pd.read_csv('Anonymous_output/S_83130/PANORAMIC_RIGHT.csv')
imu['timestamp'] = pd.to_datetime(imu['timestamp'])
imu['time_sec'] = (imu['timestamp'] - imu['timestamp'][0]).dt.total_seconds()


# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(imu['time_sec'], imu['acc_x_g'], label='acc_x_g')
plt.plot(imu['time_sec'], imu['acc_y_g'], label='acc_y_g')
plt.plot(imu['time_sec'], imu['acc_z_g'], label='acc_z_g')

# Add legend
plt.legend(loc='upper right')  

plt.xlabel('Time (s)')
plt.ylabel('Acceleration (g)')
plt.title('IMU Data with Activity Labels')
plt.grid(True)
plt.tight_layout()
plt.show()