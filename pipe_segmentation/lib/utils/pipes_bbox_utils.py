import numpy as np
import math
import os
from numpy import random
import itertools

def rigid_transform_3D(A, B):

    '''
    find the rotation matrix of rotating A to B
    Input: expects Nx3 matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 1x3 column vector
    '''
    # transpose the matrices
    A = A.T
    B = B.T
    # now they are 3xN
    
    assert A.shape == B.shape, 'The shape of corresponding bbox corners don\'t match '

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1).reshape(3,1)
    centroid_B = np.mean(B, axis=1).reshape(3,1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R@centroid_A + centroid_B

    return R.T, t.T


def make_canonical_bbox(orig_bbox, swap=False):
    '''find the bbox that is aligned to the axis with same dims as the 
    original bbox'''
    
    # only the first and last matter
    adjacancy =  [
        [0, 1,2,4],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7, 6,5,3]
        ] 
    swaped_adjacancy = [
        [0, 2,1,4],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7, 5,6,3]
        ] 
    # for some of the bboxes the order of the points of one side
    # is clockwise for one bbox and counter clock-wise for the other
    # changing the order of points 2 with 1 and 5 with 6 corrects that
    if swap:
        adjacancy = swaped_adjacancy

    # the convention below should also happen in the loop furthur down !
    # find the w,h,l
    # as a convention I call the length of corner 
    # 0 to 1 w (in the direction of x axis)
    # 0 to 2 h (y axis)
    # 0 to 4 l (z axis)
    # but if we have a swap 1 and 2 change place

    #x
    w = np.linalg.norm(orig_bbox[adjacancy[0][0]] - orig_bbox[adjacancy[0][1]])
    x_min = -w/2
    x_max = w/2

    #y
    h = np.linalg.norm(orig_bbox[adjacancy[0][0]] - orig_bbox[adjacancy[0][2]])
    y_min = -h/2
    y_max = h/2

    #z
    l =  np.linalg.norm(orig_bbox[adjacancy[0][0]] - orig_bbox[adjacancy[0][3]])
    z_min = -l/2
    z_max = l/2


    # loop over the points to find corresponding canonical ones
    canonical_bbox = np.zeros((8,3))

    # in my bboxes point 0 and 7 are always on opposite sides
    # set points 0 (all min) and 7 (all max)
    canonical_bbox[0,:] = [x_min, y_min, z_min]
    canonical_bbox[7,:] = [x_max, y_max, z_max]


    for idx in [0,7]:
        for dim_idx, neighbour_idx in enumerate(adjacancy[idx][1:]):
            distance = np.linalg.norm(orig_bbox[idx] - orig_bbox[neighbour_idx])
            # I always add the distance to x dim of 1st neighbour
            # y dim of second and z dim of third
            if idx == 0:
                # for 0 we always add
                sign = 1
            else:
                # for 7 we always subtract
                sign = -1

            canonical_bbox[neighbour_idx] = canonical_bbox[idx]
            canonical_bbox[neighbour_idx, dim_idx] += sign * distance

    return canonical_bbox, w,h,l


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB rotm2euler.m except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    '''input : The input rotation matrix must be in the post-multiply form for rotations.
    returns: the rotation about x,y,z axis in radians'''
    
    R = R.T

    assert(isRotationMatrix(R)), 'There is a problem with the rotation matrix'
    
    sy = np.sqrt(R[0,0]**2 +  R[1,0]**2)
    
    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    #the rotation about x,y,z axis in radians
    return np.array([x, y, z])


# check
# Calculates Rotation Matrix given euler angles.
def euler_Angles_To_Rotation_Matrix(theta) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R.T


def get_bbox_angles(bbox):
    orig_bbox = bbox

    # find canonical bbox
    canonical_bbox, w,h,l = make_canonical_bbox(orig_bbox)
    
    # get the rotation matrix and translation
    # rotation from canonical to original
    R, t = rigid_transform_3D(canonical_bbox, orig_bbox)
    
    # recreate the original points to see the amount of floating
    # point error
    estimated_points = canonical_bbox @ R + t

    err = np.linalg.norm((orig_bbox - estimated_points))

    if (err > 1e-9):
        # if we have too much error try switching the neighbors
        # sometimes neighbors make a right handed system and sometimes a left handed causing the problem

        # change the canonical bbox
        # swap corners 1 with 2 and 5 with 6
        canonical_bbox, w,h,l = make_canonical_bbox(orig_bbox, swap=True)
        
        # repeat the calcs
        R, t = rigid_transform_3D(canonical_bbox, orig_bbox)
        estimated_points = canonical_bbox @ R + t
        err = np.linalg.norm((orig_bbox - estimated_points))
        
        if (err > 1e-9):
            raise ValueError('rotation matrix error too high')
    
    # get the angles
    angles = rotationMatrixToEulerAngles(R)
    my_R = euler_Angles_To_Rotation_Matrix(angles)
    
    err = np.linalg.norm(my_R- R)
    
    if err > 1e-9:
        raise ValueError('angle error too high')

    
    # now we know the angles are correct return them
    return angles, w,h,l
        


def get_bbox_features_from_corners(bboxes):
    ''' make a 2D np array of features of bboxes
    input : dictionary of bboxes 
    output: 2D np array: x,y,z, w,h,l , rx,ry,rz, class_num
    '''
    all_bboxes = []

    for class_num, class_bboxes in bboxes.items():
        for bbox in class_bboxes:
            # first find center of bbox x,y,z by averaging the 8 corners
            x,y,z = np.mean(bbox, axis=0)

            # find the orientation of the box 
            angles, w,h,l  = get_bbox_angles(bbox)
            rx,ry,rz = angles

            # make numpy array and append
            all_bboxes.append(np.array([x,y,z, w,h,l, rx,ry,rz, float(class_num)])) 
    
    return np.asarray(all_bboxes)



# this is the Inverse function
def get_bbox_corners_from_features(features):
    '''
    This function returns the bbox corners that corresponds to the input features
    Input should be a list of features in order x,y,z, w,h,l, rx,ry,rz
    '''
    x,y,z, w,h,l, rx,ry,rz = features

    center = np.array([x,y,z])

    canonical_box = make_canonical_bbox_from_features(w,h,l)

    # get rotation matrix
    R = euler_Angles_To_Rotation_Matrix([rx,ry,rz])

    #reconstruct the original bbox
    reconstructed_bbox = canonical_box @ R + center

    return reconstructed_bbox

def make_canonical_bbox_from_features(w,h,l):

    x_min = -w/2
    x_max = w/2

    y_min = -h/2
    y_max = h/2

    z_min = -l/2
    z_max = l/2

    
    corners = np.asarray(list(itertools.product([x_min,x_max], [y_min,y_max], [z_min,z_max])))
    return corners
    


#This is for debug
'''import sys

if __name__ == '__main__':

    # import reader class
    sys.path.insert(1, '/cvlabdata2/home/farivar/my_network/pipe_detection/lib/datasets/')
    import pipes_dataset as reader

    # think in volumes 
    # check how much volume our boxes have in common

    # check all bboxes
    current_path = '/cvlabdata2/home/farivar/my_network/pipe_detection/tools'
    DATA_PATH = os.path.join(current_path, '../data')
    dataset =  reader.pipes_dataset(DATA_PATH, 'test')

    for i in range(dataset.num_sample):
        bboxes = dataset.get_bboxes(i)
        result = get_bbox_features_from_corners(bboxes)
        
        orig_bboxes = []
        # extract the original bboxes into a simple list
        for key, value in bboxes.items():
            for box in value:
                orig_bboxes.append(box)
        orig_bboxes = np.asarray(orig_bboxes)
        

        for bbox_idx, features in enumerate(result):
            # reconstruct the bbox
            reconstructed_corners = get_bbox_corners_from_features(features[:-1])
 
            # comapre the volume of orig_bboxes[bbox_idx] and reconstructed_corners
            # too hard didn't do !

            #instead compare the err of first and last points
            err0 = np.linalg.norm((orig_bboxes[bbox_idx] - reconstructed_corners)[0])
            err1 = np.linalg.norm((orig_bboxes[bbox_idx] - reconstructed_corners)[7])

            if (err0+err1) > 1e-10:
                # the order of error should be around 1e-15 or -16 
                raise ValueError('reconstruction error too big') 

            # save the points
            #if i%100 == 0:
            #    print()
            #    print(orig_bboxes[bbox_idx])
            #    print()
            #    print(reconstructed_corners)
            #    #np.savez('boxes', original=orig_bboxes[bbox_idx], standard=reconstructed_corners)
            #    input('waiting:')

            '''











        

                


















    