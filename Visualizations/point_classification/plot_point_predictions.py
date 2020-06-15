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
data = np.load('intact_pointnet_epoch126/9_39.npz', mmap_mode='r')
#triple_num_points_thresh_0_2_epoch_15
#pointnet_halved_0_2_as_thresh_epoch_10
#new_pointnet_halved_0_2_thresh_epoch27
#intact_pointnet_epoch126

xyz=data['points'] 
print('num of points:', len(xyz))

preds = data['segmentation']

labels = data['labels']

print('number of positive predictions', len(preds[preds>0]))

# make the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# group points into :

cases = []

# 1. true positive
true_positive_mask = (preds > 0) & (labels==preds)
true_positive = xyz[true_positive_mask]
cases.append(true_positive)

# 2. false positive
false_positve_mask = (preds>0) & (labels!=preds)
false_positve = xyz[false_positve_mask]
cases.append(false_positve)

# 3. true negative (not important)
true_negative_mask = (preds<=0) & (labels==preds)
true_negative = xyz[true_negative_mask]
cases.append(true_negative)

# 4. false negative
false_negative_mask = (preds<=0) & (labels!=preds)
false_negative = xyz[false_negative_mask]
cases.append(false_negative)

colors = ['green', 'red', 'gray', 'cyan']

names = ['true positive', 'false positive', 'true negative (not important)', 'false negative']

for case, color in zip(cases,colors):
  print(len(case))

  ax.scatter(case[:,0], case[:,1], case[:,2], marker='.', color=color, s=10)

print('confusion matrix:')
for name, case in zip(names,cases):
  print(name, ': ', len(case)/len(xyz))

performance = len(true_positive) / ( len(true_positive) + len(false_positve) + len(false_negative) )
print('Performance for this scene: %.3f'%performance)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#set_axes_equal(ax) # for equal axis

# select a section of the plot
'''plt.xlim(-1,1)
plt.ylim(-0.5,1.5)
ax.set_zlim(-2,1)'''

plt.show()