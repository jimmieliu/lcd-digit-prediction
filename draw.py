from math import cos, sin, pi
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import dataset as ds
# verts = [
#    (0.01, -0.01),  # left, bottom
#    (0.01, 0.01),  # left, top
#    (0.49, 0.01),  # right, top
#    (0.49, -0.01),  # right, bottom
#    (0, -0.01)  # ignored
# ]
_st = np.array([0, 0])
_end = np.array([0.5, 0])
_wh = np.array([0.02, 0.02])
verts = np.array(
    [
        _st + _wh * [1, -1],  # left, bottom
        _st + _wh * [1, 1],  # left, top
        _end + _wh * [-1, 1],  # right, top
        _end + _wh * [-1, -1],  # right, bottom
        _st + _wh * [1, -1],  # ignored
    ]
)

theta = -pi/2
aff_trans = np.array([
    [cos(theta), sin(theta)],
    [-sin(theta), cos(theta)],
])

# verts = verts @ aff_trans

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]


fig, ax = plt.subplots()

for i in range(10):
    translation = np.array([1*i, 0])
    for j, b in enumerate(ds.NUMBERS[i]):
        if ds.MASKS[i][j] != 1: continue
        _verts = ds.affinity_trans(verts, *b) + translation
        path = Path(_verts, codes)
        patch = patches.PathPatch(path, facecolor="orange", lw=2)
        ax.add_patch(patch)
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 2)
plt.show()
