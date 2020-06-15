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
    keep = [p for p in points if np.random.rand()<prob ]
    return np.asarray(keep)

# read corners
with open('bboxes.p', 'rb') as f:
    corners = pickle.load(f)

# read the points
points = np.load('points.npz', mmap_mode='r')
points = points['points']
# sample points
points = sample_points(points, prob=0.2)


# plot

# make the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



ax.scatter(points[:,0],points[:,1], points[:,2], marker='.', color='b', s=0.1)

for _, class_corner in corners.items():
    for corner in class_corner:
        ax.scatter(corner[:,0], corner[:,1], corner[:,2], marker='o', color='r', s=10)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

set_axes_equal(ax) # for equal axis

plt.show()