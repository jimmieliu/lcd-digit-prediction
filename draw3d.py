from math import cos, sin, pi
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import dataset3d as ds

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

_st_3d = np.array([0, 0, 0])
_end_3d = np.array([0.5, 0, 0])
_wh_3d = np.array([0.02, 0.02, 0])

verts = np.array(
    [
        _st + _wh * [1, -1],  # left, bottom
        _st + _wh * [1, 1],  # left, top
        _end + _wh * [-1, 1],  # right, top
        _end + _wh * [-1, -1],  # right, bottom
        _st + _wh * [1, -1],  # ignored
    ]
)

verts3d = np.array(
    [
        _st_3d + _wh_3d * [1, -1, 1],  # left, bottom
        _st_3d + _wh_3d * [1, 1, 1],  # left, top
        _end_3d + _wh_3d * [-1, 1, 1],  # right, top
        _end_3d + _wh_3d * [-1, -1, 1],  # right, bottom
        _st_3d + _wh_3d * [1, -1, 1],  # ignored
    ]
)

theta = -pi / 2
aff_trans = np.array(
    [
        [cos(theta), sin(theta)],
        [-sin(theta), cos(theta)],
    ]
)

# verts = verts @ aff_trans

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]


def demo(pred_trans, pred_rot, pred_mask):
    fig, ax = plt.subplots()
    for j in range(7):
        if pred_mask[j][0] > pred_mask[j][1] or pred_mask[j][1] == 0:
            continue
        _verts = ds.affinity_trans(
            verts, pred_trans[j][0], pred_trans[j][1], pred_rot[j][-1]
        )
        path = Path(_verts, codes)
        patch = patches.PathPatch(path, facecolor="orange", lw=2)
        ax.add_patch(patch)
        # print(i, "\n", _verts)
        # if j == 1: break
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 2)
    plt.show()


def demo_rot(trans, rot_mats, pred_mask):
    fig, ax = plt.subplots()
    for j in range(7):
        if pred_mask[j][0] > pred_mask[j][1] or pred_mask[j][1] == 0:
            continue

        # rot_mat = ds.get_3d_rot_mat(np.asarray(angles[j]))
        # print(rot_mat.shape)
        # print(verts3d.shape)
        # print((rot_mat @ verts3d.copy().T).shape)
        # print(trans.shape)
        _verts3d = (rot_mats[j] @ verts3d.copy().T).T + trans[j]
        # _verts3d = verts3d.copy() @ rot_mat + trans[j]
        path = Path(_verts3d[:, :2], codes)
        patch = patches.PathPatch(path, facecolor="orange", lw=2)
        ax.add_patch(patch)

        # _verts = ds.affinity_trans(verts, trans[j][0], trans[j][1], angles[j][0])
        # print(j, "\n", _verts3d)
        # if j == 1: break

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 2)
    plt.show()

def demo_angles(trans, angles, pred_mask):
    fig, ax = plt.subplots()
    for j in range(7):
        if pred_mask[j][0] > pred_mask[j][1] or pred_mask[j][1] == 0:
            continue

        rot_mat = ds.get_3d_rot_mat(np.asarray(angles[j]))
        # print(rot_mat.shape)
        # print(verts3d.shape)
        # print((rot_mat @ verts3d.copy().T).shape)
        # print(trans.shape)
        _verts3d = (rot_mat @ verts3d.copy().T).T + trans[j]
        # _verts3d = verts3d.copy() @ rot_mat + trans[j]
        path = Path(_verts3d[:, :2], codes)
        patch = patches.PathPatch(path, facecolor="orange", lw=2)
        ax.add_patch(patch)

        # _verts = ds.affinity_trans(verts, trans[j][0], trans[j][1], angles[j][0])
        # print(j, "\n", _verts3d)
        # if j == 1: break

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 2)
    plt.show()


def demo0to9():
    fig, ax = plt.subplots()
    for i in range(10):
        translation = np.array([1 * i, 0])
        for j, b in enumerate(ds.NUMBERS[i]):
            if ds.MASKS[i][j] != 1:
                continue
            _verts = ds.affinity_trans(verts, *b) + translation
            path = Path(_verts, codes)
            patch = patches.PathPatch(path, facecolor="orange", lw=2)
            ax.add_patch(patch)
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 2)
    plt.show()


if __name__ == "__main__":
    for i in range(10):
        no_feat = np.asarray(ds.NUMBERS[i])
        no_mask = np.repeat(np.asarray(ds.MASKS[i]).reshape(7, 1), repeats=2, axis=-1)
        # demo(no_feat[:, :3], no_feat[:, 3:], no_mask)
        demo_rot(no_feat[:, :3], no_feat[:, 3:], no_mask)
        # break
