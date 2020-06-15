import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from laspy.file import File

import argparse

'''parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./gt_database')
parser.add_argument('--class_name', type=str, default='Car')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()'''


def in_hull(p, hull):
    """
    return points `p` that are inside `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
        
    return p[hull.find_simplex(p)>=0]

def clean_samples(orig_lidar_file, scene_sampels_folder, bboxes_file, scene_idx):

    # read the lidar and get the points
    lidar_inFile = File(orig_lidar_file, mode = "r")
    orig_points = np.vstack((lidar_inFile.x, lidar_inFile.y, lidar_inFile.z )).T
    
    # do the same standardization that we did for finding bboxes
    scaler = StandardScaler()
    scaler.fit(orig_points)

    # get the bboxes we calculated
    # read corners
    # each key is th class number and the value is the bboxes belonging to objects of that class
    with open(bboxes_file, 'rb') as f:
        corners = pickle.load(f)

    count_sampels_saved = 0

    # go through each lidar sample and transform
    for sample_idx, sample_filename in enumerate(os.listdir(scene_sampels_folder)):
        # ignore ".ipynb_checkpoints" files
        if sample_filename == '.ipynb_checkpoints':
            continue
        
        sample_file_dir = os.path.join(scene_sampels_folder, sample_filename)

        # read the lidar sample data
        sample_inFile = File(sample_file_dir, mode = "r")
        # extract all the data needed
        sample_points = np.vstack((sample_inFile.x, sample_inFile.y, sample_inFile.z, sample_inFile.red, sample_inFile.green, sample_inFile.blue, sample_inFile.classification )).T

        # normalize the sample scenes according to the full scenes
        sample_points[:,0:3] = scaler.transform(sample_points[:,0:3])

        # check for empty bboxes (less than 10 points in a box) and don't include them in the sample's bboxes
        sample_bboxes = {}

        for class_num, class_corners in corners.items():
            # keep a list of this classe's bboxes
            class_bboxes = []

            for obj_corners in class_corners:
                # get all the points inside this bbox
                in_points = in_hull(sample_points[:,0:3], obj_corners)
                # if there is more than 10 points inside the bbox include it
                if len(in_points) >= 10:
                    class_bboxes.append(obj_corners)
            
            # if there is atleast a bbox in this scene add to the sample_bboxes
            if len(class_bboxes) >0:
                sample_bboxes[class_num] = np.asarray(class_bboxes) 

                        
        
        if len(sample_bboxes)>0:
            # if the sample is non empty save it's bboxes along side it's points
            # make a folder in final_cleaned_data the folder is coded like: str(scene_idx)+'_' + str(sample_idx) + '_' + str(sample_idx)
            # (I added sample_idx to the folder name to make sure it is unique)

            # get this files directory 
            current_path = os.path.dirname(os.path.realpath(__file__))
            folder_path = os.path.join(current_path, '../data/final_cleaned_data', str(scene_idx)+'_' + str(sample_filename[:-4]) + '_' + str(sample_idx) ) 
            os.makedirs(folder_path, exist_ok=True)

            # save the points and bboxes
            points_filename = str(scene_idx)+'_' + str(sample_filename[:-4]) + '_points'
            np.savez(os.path.join(folder_path, points_filename), points=sample_points)

            bbox_filename = str(scene_idx)+'_' + str(sample_filename[:-4]) + '_bboxes.p'
            with open(os.path.join(folder_path, bbox_filename), 'wb') as f:
                pickle.dump(sample_bboxes, f, protocol=pickle.HIGHEST_PROTOCOL)

            print('saved folder %s'%(str(scene_idx)+'_' + str(sample_filename[:-4])) )
            count_sampels_saved+=1

    return count_sampels_saved



if __name__ == '__main__':

    '''I wil perform any necessary preprocessing here
    the end result is stored in final_cleaned_data

    I need to normalize the sampled lidar data the same way the original point clouds were standardized so the bboxes would make sense.
    I also go through the bboxes and eliminate the empty or almost empty boxes.'''


    scenes_folder = ['01_jussy1',
                    '01_jussy2',
                    '01_jussy3',
                    '02_athenaz', '02_athenaz', '02_athenaz',
                    '04_tram', '04_tram', '04_tram',
                    '05_plan_les_ouates',
                    '06_champel', '06_champel',
                    '07_pictet', '07_pictet']

    scenes_lidar = ['Jussy1_rtc360_subsampled_classified_georef.las', 
               'Jussy2_rtc360_subsampled_classified_georef.las',
               'Jussy3_rtc360_subsampled_classified_georef.las',
               'Athenaz_rtc360_subsampled_classified.las', 'Athenaz_rtc360_subsampled_classified.las', 'Athenaz_rtc360_subsampled_classified.las',
               'Tram_rtc360_subsampled_classified.las', 'Tram_rtc360_subsampled_classified.las', 'Tram_rtc360_subsampled_classified.las',
               'planlesouates_rc360_subsampled_classified.las', 
               'Champel_rtc360_subsampled_classified.las','Champel_rtc360_subsampled_classified.las',
               'Pictet_rtc360_subsampled_classified.las', 'Pictet_rtc360_subsampled_classified.las']

    lidar_sample_folders = ['jussy1_sampels_avg/',
               'jussy2_sampels_avg/',
               'jussy3_sampels_avg/',
               'athenaz_sampels_sony_avg/', 'athenaz_sampels_iphone_video_avg/', 'athenaz_sampels_iphone_img_avg/',
               'tram_sampels_sony_avg/', 'tram_sampels_ktm_avg/', 'tram_sampels_iphone_video_avg/',
               'plan_ouates_sampels_avg/',
               'champel_sampels_sony_avg/', 'champel_sampels_iphone_video_avg/',
               'pictet_sampels_sony_avg/', 'pictet_sampels_iphone_video_avg/']

    bbox_folders = ['jussy1_cleaned',
                    'jussy2_cleaned',
                    'jussy3_cleaned',
                    'athenaz_cleaned', 'athenaz_cleaned', 'athenaz_cleaned',
                    'tram_cleaned', 'tram_cleaned', 'tram_cleaned',
                    'plan_ouates_cleaned',
                    'champel_cleaned', 'champel_cleaned',
                    'pictet_cleaned', 'pictet_cleaned']


    

    # currently pictet and ... have problems so they are not included
    scenes_to_exclude = [ '07_pictet']

    # get this files directory
    current_path = os.path.dirname(os.path.realpath(__file__))

    # original data path (I just made a symbolic link to the original data in the original_data folder)
    orig_data_path = os.path.join(current_path ,'../data/original_data/data')

    # lidar samples path (also a symbolic link )
    lidar_sampels_path = os.path.join(current_path ,'../data/lidar_sampels')

    # bbox path
    bbox_path = os.path.join(current_path ,'../data/bbox_data')

    total_sample_count =0

    for scene_idx, scene_folder in enumerate(scenes_folder):

        # currently pictet and ... have problems so they are not included
        if scene_folder in scenes_to_exclude:
            continue

        # get the directory of original lidar file 
        orig_lidar_file = os.path.join(orig_data_path, scene_folder, 'clouds', scenes_lidar[scene_idx])

        # get the folder of for samples of this scene
        scene_sampels_folder = os.path.join(lidar_sampels_path, lidar_sample_folders[scene_idx])

        #bboxes file
        scene_folder = os.path.join(bbox_path, bbox_folders[scene_idx])
        bboxes_file = '' 
        for file_name in os.listdir(scene_folder):
            if file_name.endswith(".p"):
                bboxes_file = os.path.join(scene_folder, file_name)

        # clean the sample scenes
        this_scenes_count_sampels = clean_samples(orig_lidar_file, scene_sampels_folder, bboxes_file, scene_idx)

        total_sample_count+= this_scenes_count_sampels

    print('total sample count:%d'%total_sample_count)

    
