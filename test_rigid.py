# test_rigid.py

import rigid
import torch

dtype = torch.float32
device = "cpu"


import numpy as np
import matplotlib.pyplot as plt


def main_bak():
    # rot = rigid.Rotation.identity(
    #     shape=(1, 3),
    #     device="cpu",
    #     dtype=torch.float32,
    # )
    # print(rot)
    # print(rot.get_quats())
    # print(rot.get_quats().shape)
    # print(rot.get_rot_mats())
    # print(rot.get_rot_mats().shape)

    # mat = rigid._get_quat("_QUAT_MULTIPLY_BY_VEC", dtype=dtype, device=device)
    # vec = torch.tensor([[1, 2, 3]])

    # print(vec.shape)
    # print(vec[..., None, :, None].shape)

    # quat = rigid.Rotation.identity(shape=vec.shape[:-1], device=device, dtype=dtype).get_quats()
    # print(quat)
    # print(quat.shape)
    # reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    # print(reshaped_mat.shape)

    # res = torch.sum(
    #     reshaped_mat *
    #     quat[..., :, None, None]  *
    #     vec[..., None, :, None],
    #     dim=(-3, -2)
    # )

    # print(res+quat)

    # print(rigid.quat_multiply_by_vec(quat, vec))

    # res = torch.tensor([[1.0, 1, 2, 3]])

    # avec = torch.tensor([[1, 1, 1.]])
    # print(rigid.quat_to_rot(res))

    # print("res", rigid.rot_vec_mul(rigid.quat_to_rot(res), avec))

    # print(rigid.quat_to_rot(res/res.square().sum().sqrt()))
    # print("res", rigid.rot_vec_mul(rigid.quat_to_rot(res/res.square().sum().sqrt()), avec))

    # print(res)
    # print(res/res.square().sum().sqrt())

    # print("res", rigid.rot_vec_mul(rigid.quat_to_rot(quat), avec))
    pass


import dataset3d as ds3d
from scipy.spatial.transform import Rotation as R


def scalar_last2first(quat):
    # if len(quat.shape) == 1:
    #     return np.array()
    return np.concatenate((quat[..., -1:], quat[..., :3]), axis=-1)


def apply_quat_to_vec(quat, vec):
    assert quat.shape[0] == 1 and quat.shape[1] == 4


def quat_cross_quat(q1, q2):
    _, b, c, d = q1
    _, f, g, h = q2
    return np.asarray((0.0, c * h - d * g, -b * h + d * f, b * g - c * f))


def quat_dot_quat(q1, q2):
    return (q1[1:] * q2[1:]).sum()


def quat_m_quat(q1, q2):
    a, b, c, d = q1
    e, f, g, h = q2
    return (
        np.asarray((a * e - quat_dot_quat(q1, q2), 0, 0, 0))
        + np.asarray((0, a, a, a)) * q2
        + np.asarray((0, e, e, e)) * q1
        + quat_cross_quat(q1, q2)
    )

def main():
    angles = np.asarray((np.pi / 2, 0, 0))
    r = R.from_euler("xyz", angles)
    print(scalar_last2first(r.as_quat()))

    r2 = R.from_euler("xyz", (np.pi * 1 / 6, 0, 0))
    r3 = R.from_euler("xyz", (np.pi * 2 / 6, 0, 0))
    q2 = scalar_last2first(r2.as_quat())
    q3 = scalar_last2first(r3.as_quat())
    print(quat_m_quat(q2, q3))

    s = 1 / q3[0]
    ts_q2 = torch.tensor(q2)
    ts_q3 = torch.tensor(q3)
    # _q3 = torch.tensor((q3))
    _q3 = torch.tensor((q3*s))
    print("|q3|", (q3**2).sum()**0.5)
    print("|_q3|", (_q3**2).sum()**0.5)
    print("q2", q2)
    print("q3", q3)
    print("_q3", _q3)
    print("l2_q3", torch.nn.functional.normalize(_q3, dim=-1))
    q4 = ts_q2 + rigid.quat_multiply_by_vec(ts_q2, _q3[1:])
    # q4 = ts_q2 + rigid.quat_multiply(ts_q2, _q3)
    print("q4", q4)
    print(q4 / (q4**2).sum()**0.5)

    q5 = rigid.quat_multiply_by_vec(ts_q2, ts_q3[1:])
    print("q5", q5)
    print(q5 / (q5**2).sum()**0.5)

    r4 = rigid.Rotation(
            rot_mats=None, 
            quats=q4, 
            normalize_quats=True,
        )
    print(r4.get_quats())

    print(rigid.quat_multiply(ts_q2, ts_q3))

def main2():
    X = np.asarray((1, 1, 1))
    Y = (1, 0, 1)
    angles = np.asarray((np.pi / 2, 0, 0))

    r = R.from_euler("xyz", angles)

    r2 = R.from_euler("xyz", (np.pi / 4, 0, 0))
    q2 = r2.as_quat() * r2.as_quat()
    print(q2 / (q2**2).sum())
    # R = ds3d.get_3d_rot_mat(angles)
    # print(R)

    Xr = (r.as_matrix() @ X.T).T
    # print(Xr)

    print(r.as_matrix())
    print(scalar_last2first(r.as_quat()))
    print("+" * 8)
    # print(r.apply(X))

    print(
        rigid.quat_to_rot(torch.as_tensor(scalar_last2first(r.as_quat())).unsqueeze(0))
    )
    print(rigid.rot_to_quat(torch.as_tensor(r.as_matrix())))

    # r2 = R.from_quat([1, 0.0, 0.0, 0.70710678])
    # print(r2.apply(X))
    print("res", rigid.rot_vec_mul(rigid.quat_to_rot(res), X.unsqueeze(0)))

    p = Plot3d()
    p.quiver(X, color="r")
    p.quiver(Xr, color="g")
    # p.quiver(X @ R, color="b")
    # p.quiver(Y, color="b")

    p.plot()


class Plot3d:
    def __init__(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        self.ax = ax

    def quiver(self, vec, color="r", st=[0, 0, 0]):
        self.ax.quiver(
            st[0],
            st[1],
            st[2],
            vec[0],
            vec[1],
            vec[2],
            color=color,
            arrow_length_ratio=0.1,
        )

    def plot(self):
        ax = self.ax
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()


if __name__ == "__main__":
    main()
