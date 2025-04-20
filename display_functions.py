import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time


def animation3D(data_pos, data_general, skip_frames = 1, time_interval = 0.001, max_size_mb = 100, size_tp = 0.5, show = True, save = False, nb_fps = 15, file_path = r"results/", file_name = r"animation3D.mp4", ffmpeg_path = None):

    if ffmpeg_path != None:
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

    Nstep = len(data_pos)
    Nframes = Nstep//skip_frames
    N_bodies = len(data_pos[0])//3
    N_tp = int(data_general[0][1])
    N_mb = N_bodies - N_tp
    tau = data_general[0][0]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    R_max = np.max(data_general[1:,1])
    plot_positions = [ax.scatter([], [], [], color='r', s = max_size_mb*data_general[i+1][1]/R_max) for i in range(N_mb)] + [ax.scatter([], [], [], color='b', s = size_tp) for i in range(N_tp)]

    max_x = np.max(np.array([np.abs(data_pos[0][3*i]) for i in range(1,N_bodies)]))
    max_y = np.max(np.array([np.abs(data_pos[0][3*i+1]) for i in range(1,N_bodies)]))
    max_z = np.max(np.array([np.abs(data_pos[0][3*i+2]) for i in range(1,N_bodies)]))

    max_ax = max([max_x, max_y, max_z])

    ax.set(xlim3d=(-max_ax, max_ax), xlabel = "x (in a.u.)")
    ax.set(ylim3d=(-max_ax, max_ax), ylabel = "y (in a.u.)")
    ax.set(zlim3d=(-max_ax, max_ax), zlabel = "z (in a.u.)")

    fig.suptitle(f"N_mb = {N_mb}, N_tp = {N_tp}, tau = {tau*365.25:.4f} days, frame_skip = {skip_frames}")

    def update_pos(frame, data_pos, plot_positions, skip_frames):

        ax.set_title(f"t = {tau*frame*skip_frames:.2f} years, frame : {frame+1}/{Nframes}")

        i = 0
        for plot_1pos in plot_positions:
            plot_1pos._offsets3d = ([data_pos[frame*skip_frames][i*3]], [data_pos[frame*skip_frames][i*3+1]], [data_pos[frame*skip_frames][i*3+2]])
            i += 1
        return plot_positions

    anim = FuncAnimation(fig, update_pos, Nframes, fargs = (data_pos, plot_positions, skip_frames), interval = time_interval, repeat = False)

    if save:
        writer = FFMpegWriter(fps=nb_fps) 
        anim.save(file_path + file_name, writer = writer)

    if show:
        plt.show()


def animation2D(data_pos, data_general, skip_frames = 1, time_interval = 0.001, max_size_mb = 100, size_tp = 0.5, show = True, save = False, nb_fps = 15, file_path = r"results/", file_name = r"animation2D.mp4", ffmpeg_path = None):

    if ffmpeg_path != None:
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

    Nstep = len(data_pos)
    Nframes = Nstep//skip_frames
    N_bodies = len(data_pos[0])//3
    N_tp = int(data_general[0][1])
    N_mb = N_bodies - N_tp
    tau = data_general[0][0]

    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    max_x = np.max(np.array([np.abs(data_pos[0][3*i]) for i in range(1,N_bodies)]))
    max_y = np.max(np.array([np.abs(data_pos[0][3*i+1]) for i in range(1,N_bodies)]))
    max_z = np.max(np.array([np.abs(data_pos[0][3*i+2]) for i in range(1,N_bodies)]))

    max_ax = 1.2*max([max_x, max_y, max_z])

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    ax1.set_xlim(-max_ax, max_ax)
    ax1.set_ylim(-max_ax, max_ax)
    ax2.set_xlim(-max_ax, max_ax)
    ax2.set_ylim(-max_ax, max_ax)

    ax1.set_xlabel("x (in a.u.)")
    ax1.set_ylabel("y (in a.u.)")
    ax2.set_xlabel("y (in a.u.)")
    ax2.set_ylabel("z (in a.u.)")

    R_max = np.max(data_general[1:,1])
    size_mb = np.array([max_size_mb*data_general[i+1][1]/R_max for i in range(N_mb)])
    color_mb = np.array([(1-0.5*(i/(N_mb-1)),0,0) for i in range(N_mb)])
    color_tp = np.array([(0,i/(N_tp-1),1) for i in range(N_tp)])
    
    plot_positions1 = np.array([ax1.scatter([], [], color = color_mb[i], s = size_mb[i]) for i in range(N_mb)] + [ax1.scatter([], [], color = color_tp[i], s = size_tp) for i in range(N_tp)])
    plot_positions2 = np.array([ax2.scatter([], [], color = color_mb[i], s = size_mb[i]) for i in range(N_mb)] + [ax2.scatter([], [], color = color_tp[i], s = size_tp) for i in range(N_tp)])

    fig.suptitle(f"N_mb = {N_mb}, N_tp = {N_tp}, tau = {tau*365.25:.4f} days, frame_skip = {skip_frames}")


    def update_pos(frame, plot_positions1, plot_positions2):

        ax1.set_title(f"t = {tau*frame*skip_frames:.2f} years")
        ax2.set_title(f"Frame : {frame+1}/{Nframes}")

        x = np.array([data_pos[frame*skip_frames][i*3] for i in range(N_bodies)])
        y = np.array([data_pos[frame*skip_frames][i*3+1] for i in range(N_bodies)])
        z = np.array([data_pos[frame*skip_frames][i*3+2] for i in range(N_bodies)])

        order_x = np.argsort(x)
        order_z = np.argsort(z)

        order = 0
        for i in order_z:
            plot_positions1[i].set_offsets([[data_pos[frame*skip_frames][i*3], data_pos[frame*skip_frames][i*3+1]]])
            plot_positions1[i].set_zorder(order)
            order += 1

        order = 0
        for i in order_x:
            plot_positions2[i].set_offsets([[data_pos[frame*skip_frames][i*3+1], data_pos[frame*skip_frames][i*3+2]]])
            plot_positions2[i].set_zorder(order)
            order += 1


    anim = FuncAnimation(fig, update_pos, Nframes, fargs = (plot_positions1, plot_positions2), interval = time_interval, repeat = False)

    if save:
        writer = FFMpegWriter(fps = nb_fps) 
        anim.save(file_path + file_name, writer = writer)

    if show:
        plt.show()



def curves(data_pos, data_general, show = True):

    Nstep = len(data_pos)
    N_bodies = len(data_pos[0])//3
    N_tp = int(data_general[0][1])
    N_mb = N_bodies - N_tp
    tau = data_general[0][0]

    plt.xlabel("Time (in years)")
    plt.ylabel("Distance from the Sun (in number of a.u.)")

    for i in range(N_mb, N_bodies):
        plt.plot(np.arange(Nstep)*tau, np.sqrt((data_pos[:,3*i] - data_pos[:,0])**2 + (data_pos[:,3*i+1] - data_pos[:,1])**2 + (data_pos[:,3*i+2] - data_pos[:,2])**2), color = (0,(i-N_mb)/max(1, N_bodies-N_mb-1),1))

    for i in range(1, N_mb):
        plt.plot(np.arange(Nstep)*tau, np.sqrt((data_pos[:,3*i] - data_pos[:,0])**2 + (data_pos[:,3*i+1] - data_pos[:,1])**2 + (data_pos[:,3*i+2] - data_pos[:,2])**2), color = (1,0.5*(i-1)/(N_mb-2),0))

    if show:
        plt.show()


def sun(data_pos, data_general, show = True):

    Nstep = len(data_pos)
    N_bodies = len(data_pos[0])//3
    N_tp = int(data_general[0][1])
    N_mb = N_bodies - N_tp
    tau = data_general[0][0]

    plt.xlabel("Time (in years)")
    plt.ylabel("Position of the Sun (in number of a.u.)")
    plt.plot(np.arange(Nstep)*tau, np.sqrt(data_pos[:,0]**2 + data_pos[:,1]**2 + data_pos[:,2]**2))

    if show:
        plt.show()