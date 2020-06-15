import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm # needed to make the axis equal
import pickle
import math


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



def sample_points(points, box_corners, prob=0.2):

    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        from scipy.spatial import Delaunay
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0


    # mask for points inside any bbox
    in_points_mask = np.full(len(points), False)
    for _, class_corner in box_corners.items():
        for corner in class_corner:
            in_points_mask += in_hull(points,corner)

    in_points = points[in_points_mask]
    out_points = points[~in_points_mask]

    # sample outpoints
    keep = [p for p in out_points if np.random.rand()<prob ]
    keep = np.asarray(keep)

    if len(keep) != 0:
        sampled_points = np.concatenate((in_points, keep), axis=0)
    else:
        sampled_points = in_points

    return sampled_points



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

def data_augmentation(aug_pts_rect, bbox_corners_dict, mustaug=False):
    """
    :param aug_pts_rect: (N, 3)
    :param aug_gt_boxes3d: (N, 7)
    :param gt_alpha: (N)
    :return: the new_aug_bboxes returned is the corners in the form of a dictionary of classes.
    """
    aug_list = ['rotation', 'scaling', 'flip']
    aug_enable = 1 - np.random.rand(3)
    if mustaug is True:
        aug_enable[0] = -1
        aug_enable[1] = -1
    aug_method = []

    AUG_METHOD_PROB = [1.0, 1.0, 0]
    if 'rotation' in aug_list and aug_enable[0] < AUG_METHOD_PROB[0]:
        # get random rotation angles 10 degrees max 
        angles = np.random.uniform(-np.pi /2, np.pi /2, size=3)
        # get the rotation matrix
        R = euler_Angles_To_Rotation_Matrix(angles)
        # each dimension is in a column so we multiply R from right
        aug_pts_rect = aug_pts_rect @ R
        aug_method.append(['rotation', angles])

    if 'scaling' in aug_list and aug_enable[1] < AUG_METHOD_PROB[1]:
        scale = np.random.uniform(0.95, 1.05)
        aug_pts_rect = aug_pts_rect * scale
        aug_method.append(['scaling', scale])

    if 'flip' in aug_list and aug_enable[2] < AUG_METHOD_PROB[2]:
        # flipping other axis can be added later
        # flip horizontal
        aug_pts_rect[:,0] = -aug_pts_rect[:,0]
        aug_method.append('flip')

    # now go throw each bbox and apply the same transforms to their corners
    # in the same order
    new_aug_bboxes = {}
    for class_num, class_bboxes in bbox_corners_dict.items():
        cur_class_bboxes = []
        for bbox in class_bboxes:
            # apply each of the transforms used
            if 'rotation' in aug_list and aug_enable[0] < AUG_METHOD_PROB[0]:
                bbox = bbox @ R
            if 'scaling' in aug_list and aug_enable[1] < AUG_METHOD_PROB[1]:
                bbox = bbox * scale
            if 'flip' in aug_list and aug_enable[2] < AUG_METHOD_PROB[2]:
                bbox[:,0] = -bbox[:,0]

            cur_class_bboxes.append(bbox)
        new_aug_bboxes[class_num] = np.asarray(cur_class_bboxes)
            

    return aug_pts_rect, new_aug_bboxes, aug_method


# read corners
with open('bboxes.p', 'rb') as f:
    corners = pickle.load(f)

# read the points
points = np.load('points.npz', mmap_mode='r')
points = points['points'][:,0:3]# only get x,y,z
# sample points
points = sample_points(points, corners, prob=0.1)


# augmentation
#points, corners, _ = data_augmentation(points,corners)

# plot

# make the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(corners)


ax.scatter(points[:,0],points[:,1], points[:,2], marker='.', color='b', s=0.1)

color = ['red', 'green', 'black', 'yellow', 'orange', 'purple', 'cyan', 'olive', 'gray', 'pink', 'brown']
i = -1
for _, class_corner in corners.items():
    i+=1
    for corner in class_corner:
        ax.scatter(corner[:,0], corner[:,1], corner[:,2], marker='o', color=color[i], s=10)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

set_axes_equal(ax) # for equal axis

# select a section of the plot
'''plt.xlim(-1,0)
plt.ylim(0,1.5)
ax.set_zlim(-1.4,-1)'''

plt.show()