import os
import numpy as np
import torch.utils.data as torch_data
import pickle



class pipes_dataset(torch_data.Dataset):

    def __init__(self, root_dir, split='train'):
        # get the final_cleaned_data folder
        self.data_dir = os.path.join(root_dir, 'final_cleaned_data')

        # do the test train split
        # I will put scene 0 to 8 as train and 9, 10, 11 as test (2 scenes plan_les_ouates and champel)
        train = set(range(9))
        test = set([9,10,11])

        split_set = train if split=='train' else test

        # get the name of the folder belonging to the split we asked for (folder names are like "scene number_id")
        self.sample_folder_list = [x for x in os.listdir(self.data_dir) if int(x.split(sep='_')[0]) in split_set ]
        self.num_sample = len(self.sample_folder_list)

    def get_lidar(self, idx):
        ''' idx is an index of the self.sample_folder_list
        
        returns : a numpy array each row a point with data like:
        x,y,z, red, green, blue, classification '''
        # get the folder name
        folder_name = self.sample_folder_list[idx]
        sample_folder = os.path.join(self.data_dir, folder_name)
        for file_name in os.listdir(sample_folder):
            if file_name.endswith(".npz"):
                points_file = os.path.join(sample_folder, file_name)

        return np.load(points_file)['points']

    def get_bboxes(self, idx):
        # get the folder name
        folder_name = self.sample_folder_list[idx]
        sample_folder = os.path.join(self.data_dir, folder_name)
        for file_name in os.listdir(sample_folder):
            if file_name.endswith(".p"):
                bboxes_file = os.path.join(sample_folder, file_name)

        with open(bboxes_file, 'rb') as f:
            bboxes = pickle.load(f)

        return bboxes


    def get_points_and_bbox (self,idx):
        # get both of them
        folder_name = self.sample_folder_list[idx]
        sample_folder = os.path.join(self.data_dir, folder_name)
        for file_name in os.listdir(sample_folder):
            if file_name.endswith(".npz"):
                points_file = os.path.join(sample_folder, file_name)
            if file_name.endswith(".p"):
                bboxes_file = os.path.join(sample_folder, file_name)
        
        points = np.load(points_file)['points']
        with open(bboxes_file, 'rb') as f:
            bboxes = pickle.load(f)

        return points, bboxes


    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError



# this is for debugging
if __name__ == '__main__':
    current_path = '/cvlabdata2/home/farivar/my_network/pipe_detection/tools'
    DATA_PATH = os.path.join(current_path, '../data')

    dataset = pipes_dataset(DATA_PATH, 'train')



    all_scenes_xyz = []
    for idx in range(len(dataset.sample_folder_list)):
        print(idx)
        #get xyz
        xyz = dataset.get_lidar(idx)[:,0:3]
        all_scenes_xyz.append(xyz)

        print(xyz.shape)

    all_xyz = np.vstack(all_scenes_xyz)
    print(all_xyz.shape)

    mean_xyz = np.mean(all_xyz,axis=0)
    print(mean_xyz.shape)

    # get all distances to center
    dists = np.sqrt(np.sum((all_xyz - mean_xyz)**2,axis=1))
    print(dists.shape)

    median_dist = np.quantile(dists, q = 0.5)

    print(median_dist)

    # the answer is 1.114457608414804