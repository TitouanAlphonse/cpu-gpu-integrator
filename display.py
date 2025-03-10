from display_functions import *


# data_general_CPU = np.loadtxt("results/general_data_CPU.txt")
# data_pos_CPU = np.loadtxt("results/positions_CPU.txt")
data_general_GPU = np.loadtxt("results/general_data_GPU.txt")
data_pos_GPU = np.loadtxt("results/positions_GPU.txt")


start = time.time()

# plt.figure("CPU")
# curves(data_pos_CPU, data_general_CPU, show = False)
plt.figure("GPU")
curves(data_pos_GPU, data_general_GPU)
# animation2D(data_pos, data_general, 10, 0.00001, 100, 1, show = True, save = False)
# animation3D(data_pos, data_general, 1, 0.00001, 100, 1, show = True, save = False)

end = time.time()

print(f"Execution time : {end - start} s")