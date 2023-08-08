from math import cos, sin, pi
import numpy as np

"""
an array of []

(x, y, z=0), ()
"""

horz = (0, 0, 0)
vert = (0, 0, -pi / 2)
"""
         --- # bar 0
bar 5 # |   | # bar 1
bar 6 #  ---
bar 4 # |   | # bar 2
         --- # bar 3
"""
BAR0 = (0, 1, 0) + horz
BAR1 = (0.5, 1, 0) + vert
BAR2 = (0.5, 0.5, 0) + vert
BAR3 = (0, 0, 0) + horz
BAR4 = (0, 0.5, 0) + vert
BAR5 = (0, 1, 0) + vert
BAR6 = (0, 0.5, 0) + horz
NOBAR = (-1, -1, -1, -1, -1 , -1)

def get_3d_rot_mat(angles):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = angles.astype(np.float32)
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R

def get_rot_mat(theta):
    return np.array(
        [
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)],
        ]
    )


def affinity_trans(vec, x, y, theta):
    return (get_rot_mat(theta) @ vec.T).T + np.array([x, y])


BAR_LEN = 0.5

NUMBERS = [
    [BAR0, BAR1, BAR2, BAR3, BAR4, BAR5],  # 0
    [BAR1, BAR2],  # 1
    [BAR0, BAR1, BAR3, BAR4, BAR6],  # 2
    [BAR0, BAR1, BAR2, BAR3, BAR6],  # 3
    [BAR1, BAR2, BAR5, BAR6],  # 4
    [BAR0, BAR2, BAR3, BAR5, BAR6],  # 5
    [BAR0, BAR2, BAR3, BAR4, BAR5, BAR6],  # 6
    [BAR0, BAR1, BAR2],  # 7
    [BAR0, BAR1, BAR2, BAR3, BAR4, BAR5, BAR6],  # 8
    [BAR0, BAR1, BAR2, BAR3, BAR5, BAR6],  # 9
]

MASKS = []

for i in range(len(NUMBERS)):
    l = len(NUMBERS[i])
    NUMBERS[i] = NUMBERS[i] + [NOBAR] * (7 - l)
    MASKS.append([1.] * l + [0.] * (7 - l))
