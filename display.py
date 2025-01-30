from display_functions import *


data_general = np.loadtxt("results/general_data.txt")
data_pos = np.loadtxt("results/positions.txt")

start = time.time()

curves(data_pos, data_general)
# animation2D(data_pos, data_general, 10, 0.00001, 100, 1, show = True, save = False)
# animation2D(data_pos, data_general, 10, 0.00001, 100, 1, show = True, save = False)

end = time.time()

print(f"Execution time : {end - start} s")