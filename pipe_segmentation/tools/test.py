import os
scene_sampels_folder = '../data/bbox_data'

current_path = os.path.dirname(os.path.realpath(__file__))


path = os.path.join(current_path, scene_sampels_folder)

for filename in os.listdir(path):
    print("\'"+ filename+"\'"+ ',')