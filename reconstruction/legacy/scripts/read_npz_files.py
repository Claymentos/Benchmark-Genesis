import numpy as np
file = '/home/cl-ment-prigent/Benchmark-Genesis/reconstruction/human_videos/grasping_cup/wuji_traj.npz'
data = np.load(file)

# List the arrays stored inside
print(data.files)

# Access individual arrays by their key name
# arr1 = data['array_name']

# Or iterate over everything
# for key in data.files:
#     print(key, data[key].shape, data[key].dtype)

# data.close()  # or use a `with` block