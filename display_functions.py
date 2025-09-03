import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def anim3D_ax(fig, ax, data_pos, data_general, skip_frames = 1, time_interval = 0.001, max_size_mb = 100, size_tp = 1, background_color = 'white', ax_display = True, grid_display = True, ax_color = 'black', save = False, nb_fps = 15, file_name = r"animation3D.mp4", ffmpeg_path = None):

    N_step = len(data_pos)
    N_frames = N_step//skip_frames
    N_bodies = len(data_pos[0]-1)//3
    N_tp = int(data_general[0][1])
    N_mb = N_bodies - N_tp
    tau = data_general[0][0]

    R_max = np.max(data_general[1:,1])
    size_mb = np.array([max_size_mb*data_general[i+1][1]/R_max for i in range(N_mb)])
    color_mb = np.array([(1, 0.9*(1-i/N_mb), 0) for i in range(N_mb)])
    color_tp = np.array([(0, i/N_tp, 1) for i in range(N_tp)])

    max_x = np.max(np.abs(data_pos[:,1::3]))
    max_y = np.max(np.abs(data_pos[:,2::3]))
    max_z = np.max(np.abs(data_pos[:,3::3]))

    ax.set_xlim(-max_x, max_x)
    ax.set_ylim(-max_y, max_y)
    ax.set_zlim(-max_z, max_z)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)

    # # ax.set_aspect('equal')

    ax.set_xlabel("x coordinate (in a.u.)")
    ax.set_ylabel("y coordinate (in a.u.)")
    ax.set_zlabel("z coordinate (in a.u.)")

    fig.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    ax.axis(ax_display)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    if ax_display and grid_display:
        ax.grid(True, color = ax_color)
        ax.xaxis.pane.set_edgecolor(ax_color)
        ax.yaxis.pane.set_edgecolor(ax_color)
        ax.zaxis.pane.set_edgecolor(ax_color)
    else:
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

    ax.xaxis.label.set_color(ax_color)
    ax.yaxis.label.set_color(ax_color)
    ax.zaxis.label.set_color(ax_color)

    ax.tick_params(colors = ax_color)
    ax.xaxis.line.set_color(ax_color)
    ax.yaxis.line.set_color(ax_color)
    ax.zaxis.line.set_color(ax_color)

    plot_positions = [ax.scatter([], [], [], color = color_mb[i], s = size_mb[i]) for i in range(N_mb)] + [ax.scatter([], [], [], color = color_tp[i], s = size_tp) for i in range(N_tp)]


    def update_pos(frame, data_pos, plot_positions, skip_frames):

        if ax_display:
            ax.set_title(f"t = {tau*data_pos[frame*skip_frames][0]:.2f} years, frame : {frame+1}/{N_frames}", color = ax_color)

        i = 0
        for plot_1pos in plot_positions:
            plot_1pos._offsets3d = ([data_pos[frame*skip_frames][i*3+1]], [data_pos[frame*skip_frames][i*3+2]], [data_pos[frame*skip_frames][i*3+3]])
            i += 1
        return plot_positions

    anim = FuncAnimation(fig, update_pos, N_frames, fargs = (data_pos, plot_positions, skip_frames), interval = time_interval, repeat = False)

    if save:
        print("Saving the animation...")
        if ffmpeg_path != None:
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        writer = FFMpegWriter(fps = nb_fps)
        anim.save("outputs/"+file_name, writer = writer)
        print("Animation well saved in outputs/"+file_name)

    return anim


def anim3D(data_pos, data_general, skip_frames = 1, time_interval = 0.001, max_size_mb = 100, size_tp = 1, background_color = 'white', ax_display = True, grid_display = True, ax_color = 'black', show = True, save = False, nb_fps = 15, file_name = r"animation3D.mp4", ffmpeg_path = None):
    fig = plt.figure("3D animation", figsize=(8,8))
    ax = fig.add_subplot(projection="3d")
    anim = anim3D_ax(fig, ax, data_pos, data_general, skip_frames, time_interval, max_size_mb, size_tp, background_color, ax_display, grid_display, ax_color, save, nb_fps, file_name, ffmpeg_path)

    if show:
        plt.show()



def anim2D_ax(fig, ax, data_pos, data_general, orientation, skip_frames = 1, time_interval = 0.001, max_size_mb = 100, size_tp = 1, background_color = 'white', ax_display = True, grid_display = True, ax_color = 'black', save = False, nb_fps = 15, file_name = r"animation3D.mp4", ffmpeg_path = None):

    N_step = len(data_pos)
    N_frames = N_step//skip_frames
    N_bodies = len(data_pos[0]-1)//3
    N_tp = int(data_general[0][1])
    N_mb = N_bodies - N_tp
    tau = data_general[0][0]

    R_max = np.max(data_general[1:,1])
    size_mb = np.array([max_size_mb*data_general[i+1][1]/R_max for i in range(N_mb)])
    color_mb = np.array([(1, 0.9*(1-i/N_mb), 0) for i in range(N_mb)])
    color_tp = np.array([(0, i/N_tp, 1) for i in range(N_tp)])

    if orientation == "x":
        offset_orientation_1 = 2
        offset_orientation_2 = 3
        offset_orientation_normal = 1
        ax.set_xlabel("y coordinate (in a.u.)", color=ax_color)
        ax.set_ylabel("z coordinate (in a.u.)", color=ax_color)
        title = "Projection on the YZ plane"
    if orientation == "y":
        offset_orientation_1 = 1
        offset_orientation_2 = 3
        offset_orientation_normal = 2
        ax.set_xlabel("x coordinate (in a.u.)", color=ax_color)
        ax.set_ylabel("z coordinate (in a.u.)", color=ax_color)
        title = "Projection on the XZ plane"
    if orientation == "z":
        offset_orientation_1 = 1
        offset_orientation_2 = 2
        offset_orientation_normal = 3
        ax.set_xlabel("x coordinate (in a.u.)", color=ax_color)
        ax.set_ylabel("y coordinate (in a.u.)", color=ax_color)
        title = "Projection on the XY plane"
    
    if orientation == "x" or orientation == "y" or orientation == "z":

        fig.set_facecolor(background_color)
        ax.set_facecolor(background_color)

        max_1 = np.max(np.abs(data_pos[:,offset_orientation_1::3]))
        max_2 = np.max(np.abs(data_pos[:,offset_orientation_2::3]))

        ax.set_xlim(-1.1*max_1, 1.1*max_1)
        ax.set_ylim(-1.1*max_2, 1.1*max_2)

        # ax.set_aspect('equal')

        ax.axis(ax_display)
        if ax_display and grid_display:
            ax.grid(True, color = ax_color, zorder = -1)
        else:
            ax.grid(False)

        ax.tick_params(colors=ax_color)
        ax.spines['bottom'].set_color(ax_color)
        ax.spines['top'].set_color(ax_color)
        ax.spines['left'].set_color(ax_color)
        ax.spines['right'].set_color(ax_color)


        plot_positions = np.array([ax.scatter([], [], color = color_mb[i], s = size_mb[i]) for i in range(N_mb)] + [ax.scatter([], [], color = color_tp[i], s = size_tp) for i in range(N_tp)])


        def update_pos(frame, data_pos, plot_positions, skip_frames):

            if ax_display:
                ax.set_title(title+f", t = {tau*data_pos[frame*skip_frames][0]:.2f} years, frame : {frame+1}/{N_frames}", color = ax_color)

            v_normal = np.array([data_pos[frame*skip_frames][3*i+offset_orientation_normal] for i in range(N_bodies)])
            order = np.argsort(v_normal)

            v_order = 0
            for i in order:
                plot_positions[i].set_offsets([[data_pos[frame*skip_frames][3*i+offset_orientation_1], data_pos[frame*skip_frames][3*i+offset_orientation_2]]])
                plot_positions[i].set_zorder(v_order)
                v_order += 1

        anim = FuncAnimation(fig, update_pos, N_frames, fargs = (data_pos, plot_positions, skip_frames), interval = time_interval, repeat = False)

        if save:
            print("Saving the animation...")
            if ffmpeg_path != None:
                plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
            writer = FFMpegWriter(fps = nb_fps) 
            anim.save(file_name, writer = writer)
            print("Animation well saved in outputs/"+file_name)

        return anim
    else:
        return None


def anim2D(data_pos, data_general, orientation, skip_frames = 1, time_interval = 0.001, max_size_mb = 100, size_tp = 1, background_color = 'white', ax_display = True, grid_display = True, ax_color = 'black', show = True, save = False, nb_fps = 15, file_name = r"animation3D.mp4", ffmpeg_path = None):
    fig = plt.figure(f"2D animation ({orientation} orientation)", figsize=(8, 8))
    ax = fig.add_subplot()
    anim = anim2D_ax(fig, ax, data_pos, data_general, orientation, skip_frames, time_interval, max_size_mb, size_tp, background_color, ax_display, grid_display, ax_color, save, nb_fps, file_name, ffmpeg_path)

    if show:
        plt.show()



def traj3D(data_pos, data_general, mb_to_display = None, tp_to_display = None, background_color = 'white', ax_display = True, grid_display = True, ax_color = 'black', show = True):

    N_bodies = len(data_pos[0]-1)//3
    N_tp = int(data_general[0][1])
    N_mb = N_bodies - N_tp

    color_mb = np.array([(1, 0.9*(1-i/N_mb), 0) for i in range(N_mb)])
    color_tp = np.array([(0, i/N_tp, 1) for i in range(N_tp)])

    fig = plt.figure("3D trajectories", figsize=(8,8))
    ax = fig.add_subplot(projection="3d")

    if ax_display:
        ax.set_title(f"Trajectories over {data_pos[-1][0]:.2f} years", color = ax_color)

    ax.set_xlabel("x coordinate (in a.u.)")
    ax.set_ylabel("y coordinate (in a.u.)")
    ax.set_zlabel("z coordinate (in a.u.)")

    fig.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    ax.axis(ax_display)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)

    if ax_display and grid_display:
        ax.grid(True, color = ax_color)
        ax.xaxis.pane.set_edgecolor(ax_color)
        ax.yaxis.pane.set_edgecolor(ax_color)
        ax.zaxis.pane.set_edgecolor(ax_color)
    else:
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

    ax.xaxis.label.set_color(ax_color)
    ax.yaxis.label.set_color(ax_color)
    ax.zaxis.label.set_color(ax_color)

    ax.tick_params(colors = ax_color)
    ax.xaxis.line.set_color(ax_color)
    ax.yaxis.line.set_color(ax_color)
    ax.zaxis.line.set_color(ax_color)

    if mb_to_display == None:
        id_mb = range(N_mb)
    else:
        id_mb = mb_to_display

    if tp_to_display == None:
        id_tp = range(N_tp)
    else:
        id_tp = tp_to_display
        

    for i in id_tp:
        i_file = i + N_mb
        ax.plot(data_pos[:,3*i_file+1], data_pos[:,3*i_file+2], data_pos[:,3*i_file+3], color = color_tp[i])

    for i in id_mb:
        ax.plot(data_pos[:,3*i+1], data_pos[:,3*i+2], data_pos[:,3*i+3], color = color_mb[i])

    # ax.set_aspect('equal')
    # ax.set_autoscalex_on(False)
    # ax.set_autoscaley_on(False)
    # ax.set_autoscalez_on(False)
    # ax.set_box_aspect([1,1,0.1])
    # ax.set_xlim(-50, 50)
    # ax.set_ylim(-50, 50)
    # ax.set_zlim(-0.1, 1)

    if show:
        plt.show()



def traj2D(data_pos, data_general, orientation, mb_to_display = None, tp_to_display = None, background_color = 'white', ax_display = True, grid_display = True, ax_color = 'black', show = True):

    N_bodies = len(data_pos[0]-1)//3
    N_tp = int(data_general[0][1])
    N_mb = N_bodies - N_tp

    color_mb = np.array([(1, 0.9*(1-i/N_mb), 0) for i in range(N_mb)])
    color_tp = np.array([(0, i/N_tp, 1) for i in range(N_tp)])
    
    fig = plt.figure(f"2D animation ({orientation} orientation)", figsize=(8,8))
    ax = fig.add_subplot()

    if orientation == "x":
        offset_orientation_1 = 2
        offset_orientation_2 = 3
        if ax_display:
            ax.set_xlabel("y coordinate (in a.u.)", color = ax_color)
            ax.set_ylabel("z coordinate (in a.u.)", color = ax_color)
            ax.set_title(f"Projection on the YZ plane, trajectories over {data_pos[-1][0]:.2f} years", color = ax_color)
    if orientation == "y":
        offset_orientation_1 = 1
        offset_orientation_2 = 3
        if ax_display:
            ax.set_xlabel("x coordinate (in a.u.)", color = ax_color)
            ax.set_ylabel("z coordinate (in a.u.)", color = ax_color)
            ax.set_title(f"Projection on the XZ plane, trajectories over {data_pos[-1][0]:.2f} years", color = ax_color)
    if orientation == "z":
        offset_orientation_1 = 1
        offset_orientation_2 = 2
        if ax_display:
            ax.set_xlabel("x coordinate (in a.u.)", color = ax_color)
            ax.set_ylabel("y coordinate (in a.u.)", color = ax_color)
            ax.set_title(f"Projection on the XY plane, trajectories over {data_pos[-1][0]:.2f} years", color = ax_color)

    if mb_to_display == None:
        id_mb = range(N_mb)
    else:
        id_mb = mb_to_display

    if tp_to_display == None:
        id_tp = range(N_tp)
    else:
        id_tp = tp_to_display

    if orientation == "x" or orientation == "y" or orientation == "z":

        fig.set_facecolor(background_color)
        ax.set_facecolor(background_color)

        ax.axis(ax_display)
        if ax_display and grid_display:
            ax.grid(True, color = ax_color, zorder = -1)
        else:
            ax.grid(False)

        ax.tick_params(colors=ax_color)
        ax.spines['bottom'].set_color(ax_color)
        ax.spines['top'].set_color(ax_color)
        ax.spines['left'].set_color(ax_color)
        ax.spines['right'].set_color(ax_color)

        for i in id_tp:
            i_file = i + N_mb
            ax.plot(data_pos[:,3*i_file+offset_orientation_1], data_pos[:,3*i_file+offset_orientation_2], color = color_tp[i])

        for i in id_mb:
            ax.plot(data_pos[:,3*i+offset_orientation_1], data_pos[:,3*i+offset_orientation_2], color = color_mb[i])

    # ax.set_aspect('equal')

    if show:
        plt.show()



def traj1D(data_pos, data_general, orientation, mb_to_display = None, tp_to_display = None, fun_color_mb = None, fun_color_tp = None, show = True):
    
    N_bodies = len(data_pos[0]-1)//3
    N_tp = int(data_general[0][1])
    N_mb = N_bodies - N_tp
    tau = data_general[0][0]
    step = data_pos[:,0]

    if fun_color_mb == None:
        color_mb = np.array([(1, 0.9*(1-i/N_mb), 0) for i in range(N_mb)])
    else:
        color_mb = np.array([fun_color_mb(i, N_mb) for i in range(N_mb)])
    if fun_color_tp == None:
        color_tp = np.array([(0, i/N_tp, 1) for i in range(N_tp)])
    else:
        color_tp = np.array([fun_color_tp(i, N_tp) for i in range(N_tp)])

    plt.figure(f"1D trajectories (projection on {orientation})")

    if orientation == "x" or orientation == "y" or orientation == "z":
        plt.title(f"Projection on the {orientation} axis")
        plt.ylabel(f"{orientation} coordinate (in a.u.)")

    plt.xlabel("Time (in years)")
    if orientation == "x":
        offset_orientation = 1
    if orientation == "y":
        offset_orientation = 2
    if orientation == "z":
        offset_orientation = 3
    if orientation == "r":
        plt.title("Distance from the Sun")
        plt.ylabel("Distance from the Sun (in a.u.)")


    if mb_to_display == None:
        id_mb = range(N_mb)
    else:
        id_mb = mb_to_display

    if tp_to_display == None:
        id_tp = range(N_tp)
    else:
        id_tp = tp_to_display

    for i in id_tp:
        i_file = i + N_mb
        if orientation == "x" or orientation == "y" or orientation == "z":
            plt.plot(step*tau, data_pos[:,3*i_file+offset_orientation], color = color_tp[i])
        if orientation == "r":
            plt.plot(step*tau, np.sqrt((data_pos[:,3*i_file+1] - data_pos[:,1])**2 + (data_pos[:,3*i_file+2] - data_pos[:,2])**2 + (data_pos[:,3*i_file+3] - data_pos[:,3])**2), color = color_tp[i])
    
    for i in id_mb:
        if orientation == "x" or orientation == "y" or orientation == "z":
            plt.plot(step*tau, data_pos[:,3*i+offset_orientation], color = color_mb[i])
        if orientation == "r":
            if i > 0:
                plt.plot(step*tau, np.sqrt((data_pos[:,3*i+1] - data_pos[:,1])**2 + (data_pos[:,3*i+2] - data_pos[:,2])**2 + (data_pos[:,3*i+3] - data_pos[:,3])**2), color = color_mb[i])

    plt.grid(True)

    if show:
        plt.show()


def orb_param(data_orb_param, data_general, param, mb_to_display = None, tp_to_display = None, fun_color_mb = None, fun_color_tp = None, show = True):
    
    N_bodies = len(data_orb_param[0]-1)//6
    N_tp = int(data_general[0][1])
    N_mb = N_bodies - N_tp
    tau = data_general[0][0]
    step = data_orb_param[:,0]

    if fun_color_mb == None:
        color_mb = np.array([(1, 0.9*(1-i/N_mb), 0) for i in range(N_mb)])
    else:
        color_mb = np.array([fun_color_mb(i, N_mb) for i in range(N_mb)])
    if fun_color_tp == None:
        color_tp = np.array([(0, i/N_tp, 1) for i in range(N_tp)])
    else:
        color_tp = np.array([fun_color_tp(i, N_tp) for i in range(N_tp)])

    if param == "a":
        offset = 1
        plt.figure(f"Orbital parameters (a)")
        plt.ylabel(r"Semi-major axis $a$ (in a.u.)")
    if param == "e":
        offset = 2
        plt.figure(f"Orbital parameters (e)")
        plt.ylabel(r"Eccentricity $e$ (no dimension)")
    if param == "i":
        offset = 3
        plt.figure(f"Orbital parameters (i)")
        plt.ylabel(r"Inclination $i$ (in °)")
    if param == "Omega":
        offset = 4
        plt.figure(f"Orbital parameters (Omega)")
        plt.ylabel(r"Longitude of ascending node $\Omega$ (in °)")
    if param == "omega":
        offset = 5
        plt.figure(f"Orbital parameters (omega)")
        plt.ylabel(r"Argument of periapsis $\omega$ (in °)")
    if param == "M":
        offset = 6
        plt.figure(f"Orbital parameters (M)")
        plt.ylabel(r"Mean anomaly $M$ (in °)")

    plt.xlabel("Time (in yr)")

    if mb_to_display == None:
        id_mb = range(N_mb)
    else:
        id_mb = mb_to_display

    if tp_to_display == None:
        id_tp = range(N_tp)
    else:
        id_tp = tp_to_display

    for i in id_tp:
        i_file = i + N_mb
        plt.plot(step*tau, data_orb_param[:,6*i_file+offset], color = color_tp[i])
    
    for i in id_mb:
        plt.plot(step*tau, data_orb_param[:,6*i+offset], color = color_mb[i])

    plt.grid(True)
    plt.legend()

    if show:
        plt.show()