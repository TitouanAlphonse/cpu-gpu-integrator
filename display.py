from display_functions import *


data_general_CPU = np.loadtxt("results/general_data_CPU.txt")
data_pos_CPU = np.loadtxt("results/positions_CPU.txt")
# data_general_GPU_multi_t = np.loadtxt("results/general_data_GPU_multi_t.txt")
# data_pos_GPU_multi_t = np.loadtxt("results/positions_GPU_multi_t.txt")
# data_general_GPU = np.loadtxt("results/general_data_GPU.txt")
# data_pos_GPU = np.loadtxt("results/positions_GPU.txt")


start = time.time()

plt.figure("CPU")
curves(data_pos_CPU, data_general_CPU, show = False)
# plt.figure("CPU Sun")
# sun(data_pos_CPU, data_general_CPU, show = False)
# plt.figure("GPU")
# curves(data_pos_GPU, data_general_GPU, show = False)

# plt.figure("GPU_multi_t")
# curves(data_pos_GPU_multi_t, data_general_GPU_multi_t, show = False)
# plt.figure("GPU", figsize=(10,6))
# curves(data_pos_GPU, data_general_GPU)
# animation2D(data_pos, data_general, 10, 0.00001, 100, 1, show = True, save = False)
# animation3D(data_pos_CPU, data_general_CPU, 1, 0.00001, 100, 1, show = True, save = False)

plt.show()

end = time.time()

print(f"Execution time : {end - start} s")