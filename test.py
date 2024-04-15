# import os
# dir_path = './data/facebank/les/'
# count = 0
# # Iterate directory
# for path in os.listdir(dir_path):
#     if '.jpg' in path:
#         count += 1
# print(count)

# #compact version
# print(len([path for path in os.listdir(dir_path) if '.jpg' in path]))
import numpy as np
a = []
b = np.array(a)

print(b is None)
print(b.size)
print(b.size == 0)