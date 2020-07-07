
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from hyper_params import *

path = 'guppy_data/couzin_torus/train/'
pred_path = 'guppy_data/predictions_simulated/'
pred_name = 'predicted_path.npy'
path = pred_path
pred = np.load(pred_path + pred_name)
print(pred.shape) #(4, 748, 2)
print(pred)
timesteps = 748
n_fish = 0

if live_data:
    path = 'guppy_data/live_female_female/train/'
    timesteps = 8990 #tot timesteps: 8990

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


#data = np.array(data)
data = pred
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

    return start_x, end_x, start_y, end_y

def get_tips(vec_start_x, vec_end_x, vec_start_y, vec_end_y):
    vec_length = math.dist((vec_start_x, vec_start_y), (vec_end_x, vec_end_y))
    adj_length = vec_length / 3
    theta = 30
    opp_length = adj_length * math.tan(theta)
    tip_left_x,  tip_right_x, tip_left_y, tip_right_y = vec_end_x - opp_length, vec_end_x + opp_length, 0, 0

    return tip_left_x, tip_left_y, tip_right_x, tip_right_y

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
        vec_start_x, vec_end_x, vec_start_y, vec_end_y = get_vectors(frame,i)
        curr_vec =  np.array(vec_end_x,vec_end_y) - np.array(vec_start_x, vec_start_y)
        if False: #frame > 1:
            prev_start_x, prev_end_x, prev_start_y, prev_end_y = get_vectors(frame - 1, i)
            prev_vec = np.array(prev_end_x,prev_end_y) - np.array(prev_start_x,prev_start_y)
            length = math.dist((prev_start_x,prev_start_y), (prev_end_x,prev_end_y))
            ang_diff = np.arccos(np.dot(curr_vec, prev_vec) / length)
            print(ang_diff)

        #tip_left_x, tip_left_y, tip_right_x, tip_right_y = \
        #   get_tips(vec_start_x, vec_end_x, vec_start_y, vec_end_y)

        #if i % 3 == 0:
        line.set_data([vec_start_x, vec_end_x], [vec_start_y, vec_end_y])

        #if i % 3 == 1:
         #   line.set_data([vec_end_x, tip_left_x], [vec_end_y, tip_left_y])
        #else:
         #   line.set_data([vec_end_x, tip_right_x], [vec_end_y, tip_right_y])

    # can't return both orientation vectors and position dots for some reason
    #return lines
    return scat,

def plot_trajectories():
    plt.plot(pos[:, :timesteps, 0], pos[:, :timesteps, 1])
    for i in range(n_fish):
        plt.plot(pos[i, :timesteps, 0], pos[i, :timesteps, 1])

ani = animation.FuncAnimation(fig, update, frames=timesteps, interval = 10, init_func=init, blit = True, repeat = False)
plot_trajectories()
plt.show()
