# test_repr6.py
import torch
from torch import nn
import numpy as np
from scipy.spatial.transform import Rotation as R
import rigid

def normalize_vector(v):
    # batch = v.shape[0]  # b, r, 3
    v_mag = torch.sqrt(v.pow(2).sum(-1, keepdims=True))  # b, r,
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(
            torch.device("cpu")
        )
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(
            torch.device("cuda:%d" % gpu)
        )
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag  # .view(batch,1).expand(batch,v.shape[1])
    v = v / v_mag
    return v

# u, v batch*n
def cross_product(u, v):
    # batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    out = torch.cat(
        (i.unsqueeze(-1), j.unsqueeze(-1), k.unsqueeze(-1)), -1
    )  # batch*3

    return out

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[..., 0:3]  # b, r, 3
    y_raw = poses[..., 3:6]  # b, r, 3

    x = normalize_vector(x_raw)  # b, r, 3
    z = cross_product(x, y_raw)  # b, r, 3
    z = normalize_vector(z)  # b, r, 3
    y = cross_product(z, x)  # b, r, 3

    x = x.unsqueeze(-1)  # view(-1,3,1) # b, r, 3, 1
    y = y.unsqueeze(-1)  # view(-1,3,1) # b, r, 3, 1
    z = z.unsqueeze(-1)  # view(-1,3,1) # b, r, 3, 1
    matrix = torch.cat((x, y, z), -1)  # b, r, 3, 3
    return matrix


class Net(nn.Module):
    def __init__(self, fan_in, fan_out) -> None:
        super().__init__()

        self.lin = nn.Linear(fan_in, fan_out, init="final")

    def forward(self, x):
        repr6 = self.lin(x)
        rot_mat = compute_rotation_matrix_from_ortho6d(repr6)
        return rot_mat


def scalar_last2first(quat):
    # if len(quat.shape) == 1:
    #     return np.array()
    return np.concatenate((quat[..., -1:], quat[..., :3]), axis=-1)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, len):
        self.len = len

    def __getitem__(self, idx):
        euler_angles = np.asarray((np.pi / 2, 0, 0))
        r = R.from_euler("xyz", euler_angles)
        vec1 = np.random.random((3))
        vec2 = (r.as_matrix() @ vec1.T).T

        return {
            "v1": vec1,
            "init_rot_mat": R.from_euler(
                "xyz", np.asarray((np.pi / 4, 0, 0))
            ).as_matrix(),
            "target": vec2,
            "target_quat": scalar_last2first(r.as_quat()),
        }

    def __len__(self):
        return self.len


def main():
    bs = 3*128
    ds = Dataset(bs)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs)
    n = Net(3, 6)
    # opt = torch.nn.optim.Adam(n.parameters(), lr=1e-2)
    for d in dl:
        v1 = d["v1"]
        t = d["target"]
        init_rot_mat = d["init_rot_mat"]
        print(init_rot_mat.shape)
        init_rot_mat = d["init_rot_mat"].reshape(3,128,3,3)
        rot_mat = n(torch.ones((3, 128, 3)))
        rot_mat.retain_grad()
        # print(rot_mat.shape)
        # print(init_rot_mat.shape)
        final_rot_mat = rigid.rot_matmul(init_rot_mat, rot_mat)
        final_rot_mat.retain_grad()
        # loss = ((final_rot_mat @ v1.T).T - t) ** 2
        # loss.sum().backward()
        pquat = rigid.rot_to_quat(final_rot_mat)
        pquat.retain_grad()
        loss = (pquat - d["target_quat"].reshape(3,128,4)) ** 2
        loss.sum().backward()

        print(f"final_rot_mat.grad={final_rot_mat.grad}")
        print(f"rot_mat.grad={rot_mat.grad}")
        print(f"n.lin.weight.grad={n.lin.weight.grad}")
        print(f"pquat.grad={pquat.grad}")


if __name__ == "__main__":
    main()
