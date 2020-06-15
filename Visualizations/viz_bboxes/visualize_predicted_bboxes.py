import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm # needed to make the axis equal
import itertools
import math

# 9_19 num of 17 gt_bboxes

def nms(boxes, thresh, max_num_boxes=100):
    '''perform non maximum supression to select promissing bboxes for 1 scene
    bbox_features: (N,9) already sorted descending according to the classification score
    thresh: the radius from the current box centers that we use to exclude in nms
    max_num_boxes: how many boxes do you want to get (at most)
    '''
    top_proposals = boxes

    # perform the NMS
    # save the bboxes we want to return here
    # we start with the highest scoring bbox
    selected_bboxes = [top_proposals[0]] 
    for box in top_proposals:
        promissing = True
        for selected_box in selected_bboxes:
            # calculate the distance between the center of two bbboxes
            # if they are too close the current box is not promissing
            distance_centers = np.linalg.norm(box[0:3]-selected_box[0:3])
            if distance_centers < thresh:
                promissing = False
                break
        if promissing:
            selected_bboxes.append(box)
        # if we have all the boxes we asked for finish it (100 boxes max)
        if len(selected_bboxes) >= max_num_boxes:
            break
    
    # note the returned bboxes might be less than 100
    selected_bboxes = np.asarray(selected_bboxes)  
    return selected_bboxes

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

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    import scipy
    from scipy.spatial import Delaunay
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag



# read the data
data = np.load('pointnet_intact_epoch1/9_19_bboxes.npz')
boxes = data['bboxes']
points = data['points']

# try some of the thresholds and choose one
'''threshes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for t in threshes:
    selected_bboxes = nms(boxes, thresh=t)
    print()
    print('thresh: ', t)
    print(len(selected_bboxes))'''


selected_bboxes = nms(boxes, thresh=0.4)

# make the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# keep the mask of points that belong to any of the boxes (there can be repetition)
all_in_points_mask = np.zeros(len(points),  dtype=bool)

# plot the points inside each bounding box with a random color
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan']
#colors = []
for idx, box in enumerate(selected_bboxes):
    # get the corners
    box_corners = get_bbox_corners_from_features(box)

    #get the mask of points inside the box
    mask = in_hull(points, box_corners)

    all_in_points_mask += mask

    in_points = points[mask]

    # plot
    # choose a color
    if idx >= len(colors):
        color = tuple(np.random.rand(3,))
    else:
        color = colors[idx]

    # plot points in the box
    ax.scatter(in_points[:,0], in_points[:,1], in_points[:,2], marker='.', color=color, s=10)
    # plot box
    #ax.scatter(box_corners[:,0], box_corners[:,1], box_corners[:,2], marker='o', color = color, s=20)


# plot outside points
out_points = points[~all_in_points_mask]
ax.scatter(out_points[:,0], out_points[:,1], out_points[:,2], marker='.', color='gray', s=10)




ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# select a section of the plot
'''plt.xlim(-1,1)
plt.ylim(-0.5,1.5)
ax.set_zlim(-2,1)'''

plt.show()










