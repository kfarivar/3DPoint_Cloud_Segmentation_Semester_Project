import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm # needed to make the axis equal
import pickle


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


def sample_points(points, prob=0.2):
    mask =  np.random.rand(len(points)) < prob
    keep = points[mask] 
    return keep

# read corners
# /home/kiyarash/Desktop/semester_project/bounding_boxes/cleaned_data/champel_cleaned/champel_bboxes.p
with open('cleaned_data/champel_cleaned/champel_bboxes.p', 'rb') as f:
    corners = pickle.load(f)

# read the points
# /home/kiyarash/Desktop/semester_project/bounding_boxes/cleaned_data/champel_cleaned/cleaned_champel_points.npy
points = np.load('cleaned_data/champel_cleaned/cleaned_champel_points.npy', mmap_mode='r')

# select points subset
'''mask = (points[:,0] > 1.5) & (points[:,1]<-1)
points = points[mask]'''

points = sample_points(points, prob=0.5)



# color
instance_labels = []

for class_num, class_corners in corners.items():
    for i in range(len(class_corners)):
        instance_labels.append([int(class_num), i])

colors = [np.random.rand(3,) for l in instance_labels]


# plot

# make the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for inst, col in zip(instance_labels, colors):

    # select the points
    mask = (points[:,6]== inst[0]) & (points[:,7]==inst[1])
    xyz = points[mask]

    # select corner
    inst_corners = corners[str(inst[0])][inst[1]]

    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], marker='.', color=tuple(col), s=1)

    ax.scatter(inst_corners[:,0], inst_corners[:,1], inst_corners[:,2], marker='o', color=tuple(col), s=20)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

set_axes_equal(ax) # for equal axis

# select a section of the plot
plt.xlim(-1,0)
plt.ylim(0.5,1.35)
ax.set_zlim(-1.7,-0.3)

plt.show()