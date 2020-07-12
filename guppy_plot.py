
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

live_data = True
path = 'guppy_data/couzin_torus/train/'
timesteps = 750
n_fish = 2

if live_data:
    path = 'guppy_data/live_female_female/train/'
    timesteps = 2000 #tot timesteps: 8990

arr = os.listdir(path=path)
arr = arr[-2:-1] # single index doesn't work for some reason
n_files = len(arr)

data = list()
print(arr)

for filename in arr:
    #n_fish = int(filename[0]) # all fish, only for simulated data
    for fish in range(n_fish):
        with h5py.File(path + filename, "r") as f:
            key = list(f.keys())[fish]
            data.append(list(f[key]))


data = np.array(data)
pos = data[:, :, 0:2]  # position vector
x_pos = pos[:,:,0]
y_pos = pos[:,:,1]
ori = data[:, :, 2:] #orientation vector
x_ori = ori[:, :, 0]
y_ori = ori[:, :, 1]

def get_vectors(frame,i):
    start_x = pos[i, frame, 0]
    start_y = pos[i, frame, 1]
    end_x = start_x + 3 * ori[i, frame, 0]
    end_y = start_y + 3 * ori[i, frame, 1]

    return start_x,end_x, start_y, end_y

# animation

fig, ax = plt.subplots()
scat = ax.scatter([], [], animated=True)

scat.set_offsets([])

lines=[]
for index in range(n_fish):
    lobj = ax.plot([], [], animated=True)[0]
    lines.append(lobj)

def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    for line in lines:
        line.set_data([],[])
    return lines

def update(frame):
    scat.set_offsets(pos[:, frame, :])

    for i, line in enumerate(lines):
        start_x, end_x, start_y, end_y = get_vectors(frame,i)
        line.set_data([start_x, end_x], [start_y, end_y])

    # can't return both orientation vectors and position dots for some reason
    #return lines
    return scat,

def plot_trajectories():
    for i in range(n_fish):
        plt.plot(pos[i, :timesteps, 0], pos[i, :timesteps, 1])

ani = animation.FuncAnimation(fig, update, frames=timesteps, interval = 20, init_func=init, blit = True, repeat = False)
plot_trajectories()
plt.show()
