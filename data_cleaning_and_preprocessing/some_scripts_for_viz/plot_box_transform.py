import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm # needed to make the axis equal


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


# read the data
data = np.load('bboxes.npz', mmap_mode='r')

original = data['original']
standard = data['standard']

# make the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(original[:,0], original[:,1], original[:,2], marker='o', color='r', s=20)

ax.scatter(standard[:,0], standard[:,1], standard[:,2], marker='o', color='b', s=15)

for i, txt in enumerate(original):
    ax.text(original[i,0], original[i,1], original[i,2],  '%s' % (str(i)), size=20, zorder=1, color='k')

for i, txt in enumerate(standard):
    ax.text(standard[i,0], standard[i,1], standard[i,2],  '%s' % (str(i)), size=20, zorder=1, color='k')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

set_axes_equal(ax) # for equal axis

plt.show()
