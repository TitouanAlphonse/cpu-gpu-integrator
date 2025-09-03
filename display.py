from display_functions import *

suffix = ""

data_pos = np.loadtxt(f"outputs/positions{suffix}.txt")
data_orb_param = np.loadtxt(f"outputs/orb_param{suffix}.txt")
data_general = np.loadtxt(f"outputs/general_data{suffix}.txt")


r_max = 1

skip_frames = 100
time_interval = 100
max_size_mb = 100
size_tp = 1
background_color = "black"
ax_color = "white"
ax_display = True
grid_display = False

nb_fps = 60
file_name = "animation3D.mp4"

show = True
save = False


# Example of functions that can be used (uncomment the wanted function) :

# anim3D(data_pos, data_general, skip_frames, time_interval, max_size_mb, size_tp, background_color, ax_display, grid_display, ax_color, show, save, nb_fps, file_name)

# anim2D(data_pos, data_general, "y", skip_frames, time_interval, max_size_mb, size_tp, background_color, ax_display, grid_display, ax_color, show, save, nb_fps, file_name)

# traj3D(data_pos, data_general, background_color = background_color, ax_display = ax_display, grid_display = grid_display, ax_color = ax_color, show = True)

# traj2D(data_pos, data_general, "z", [2], background_color = background_color, ax_display = ax_display, grid_display = grid_display, ax_color = ax_color, show = True)

traj1D(data_pos, data_general, "r", show = True)

# orb_param(data_orb_param, data_general, "a", show = True)