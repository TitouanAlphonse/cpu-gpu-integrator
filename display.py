from display_functions import *

data_general = np.loadtxt("outputs/general_data.txt")
data_pos = np.loadtxt("outputs/positions.txt")


start = time.time()

curves(data_pos, data_general, show = False)
# sun(data_pos, data_general, show = False)
# animation2D(data_pos, data_general, 10, 0.00001, 100, 3, show = True, save = False)
# animation3D(data_pos, data_general, 10, 0.00001, 100, 1, show = True, save = False)

plt.show()

end = time.time()

print(f"Execution time : {end - start} s")