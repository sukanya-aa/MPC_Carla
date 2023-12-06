import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
path = 'myData'
cols = ['time', 'steer', 'speed', 'brake', 'throttle']
data = pd.read_csv(os.path.join(path, 'runlog1.csv'), names=cols)

time = data['time']
steer = data['steer']
speed = data['speed']

# initial conditions
x = 0
y = 0
theta = math.radians(90)  # current heading angle is 90
length = 3  # assumed length of vehicle is 3 m

# trajectory of the vehicle from center of front axle
traj_x = []
traj_y = []
for i in range(1, len(steer)):
    dt = time[i] - time[i-1]
    delta_x = speed[i] * np.cos(math.radians(steer[i]) + theta) * dt
    delta_y = speed[i] * np.sin(math.radians(steer[i]) + theta) * dt
    d_theta = (speed[i] * np.sin(math.radians(steer[i])))/length * dt

    x += delta_x
    y += delta_y
    theta += d_theta

    traj_x.append(x)
    traj_y.append(y)

plt.plot(traj_x, traj_y)
plt.title("Vehicle trajectory")
plt.xlabel("X coords")
plt.ylabel("Y coords")
plt.show()