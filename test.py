import os
dir_path = './data/facebank/selena'
# count = 0
# # Iterate directory
# for path in os.listdir(dir_path):
#     if '.jpg' in path:
#         count += 1
# print(count)

# #compact version
print(len([path for path in os.listdir(dir_path) if '.jpg' in path]))
name = 'selena'
n_files = len([path for path in os.listdir(dir_path) if '.jpg' in path])
path_img = dir_path + '/' + name +'_'+ str(n_files + 1).zfill(3) + '.jpg'
print(path_img)