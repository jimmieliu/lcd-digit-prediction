from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# import rigid
import torch
from draw3d import demo_rot

save = "model.pt"


class MLP(nn.Module):
    def __init__(self, fin, *fo_list) -> None:
        super().__init__()
        layers = [nn.LayerNorm(fin)]
        fi = fin
        for i, fo in enumerate(fo_list):
            layers.append(nn.Linear(fi, fo))
            if i < len(fo_list) - 1:
                layers.append(nn.ReLU())
            fi = fo

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        return self.layers(x)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.emb = nn.Embedding(10, 128)
        self.mlp = MLP(128, 64)

        self.act_fn = nn.ReLU()
        self.trans_net = MLP(64, 7 * 3)
        self.vec6d_net = MLP(64, 7 * 6)
        self.mask_net = MLP(64, 7 * 2)

    def get_rotation_matrix(self, vec6d):
        c1 = vec6d[..., :3]
        c2 = vec6d[..., 3:]
        # print(c1.shape, c2.shape)
        c3 = torch.linalg.cross(c1, c2)
        return torch.stack([c1, c2, c3], axis=-1)

    def get_rotated(self, repr_6d):
        init_v = tf.constant(INIT_AXES, dtype=tf.float32)
        Rs = self.get_rotation_matrix(repr_6d)
        y_pred = tf.transpose(tf.matmul(Rs, tf.transpose(init_v)), [0, 2, 1])
        return y_pred

    def dot(self, a, b):
        return (a * b).sum(axis=-1, keepdims=True)

    def forward(self, input):
        # input  # (B, L)
        x = self.emb(input)  # (B, L, D)
        x = self.mlp(x)  # (B, L, D)
        # x = self.xy_projection(x)
        B, L, _ = x.shape
        # rigids = rigid.Rigid.identity(
        #     shape=(B, L, 8),
        #     dtype=x.dtype,
        #     device=x.device,
        #     requires_grad=self.training,
        #     fmt="quat"
        # )
        x = self.act_fn(x)

        repr6d = self.vec6d_net(x).reshape(B, 7, 6)
        c1 = torch.nn.functional.normalize(repr6d[..., :3], dim=-1)
        c2 = torch.nn.functional.normalize(
            repr6d[..., 3:] - self.dot(c1, repr6d[..., 3:]) * c1, dim=-1
        )

        return self.trans_net(x), torch.concat([c1, c2], axis=-1), self.mask_net(x)


def train():
    train_iters = 20000
    log_interval = 4000
    lr = 1e-3
    model = Net()
    print(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    import dataset3d as ds
    import random

    cls_loss_fn = torch.nn.CrossEntropyLoss()
    losssum = 0
    for step in range(train_iters + 1):
        inputs = []
        target_trans = []
        target_rots = []
        masks = []
        B = 16
        for b in range(B):
            no = random.randint(0, 9)
            inputs.append([no])
            trans_angles = np.asarray(ds.NUMBERS[no])
            target_trans.append(trans_angles[:, :3])
            target_rots.append([])
            for trans_angles in ds.NUMBERS[no]:
                target_rots[-1].append(ds.get_3d_rot_mat(np.asarray(trans_angles[3:])))
            masks.append(ds.MASKS[no])
        inputs = torch.tensor(inputs)
        target_trans = torch.tensor(target_trans)
        target_rots = torch.tensor(target_rots)
        masks = torch.tensor(masks, dtype=torch.float32)

        pred_trans, repr6d, pred_mask = model(inputs)
        pred_rot_mat = model.get_rotation_matrix(repr6d.reshape(B, 7, 6))

        trans_loss = (pred_trans.reshape(B, 7, 3) - target_trans.reshape(B, 7, 3)).sum(
            axis=-1
        )  # [:, :, :2]
        trans_loss = ((masks * trans_loss).abs()).sum()

        rot_loss = (pred_rot_mat.reshape(B, 7, 9) - target_rots.reshape(B, 7, 9)).sum(
            axis=-1
        )

        rot_loss = ((masks * rot_loss) ** 2).sum()

        # print(mask.shape)
        cls_loss = cls_loss_fn(
            pred_mask.reshape(B * 7, 2), masks.reshape(B * 7).to(torch.int64)
        )

        loss = trans_loss * 10 + rot_loss + cls_loss
        losssum += loss.detach().item()

        opt.zero_grad()
        loss.backward()
        opt.step()
        # for g in opt.param_groups:
        #     if step < 100:
        #         g["lr"] = lr * min(1, step / 100)
        #     elif step < train_iters * 0.9:
        #         g["lr"] = lr
        #     else:
        #         g["lr"] = lr * min(1, (train_iters - step) / (train_iters * 0.1))

        if step % log_interval == 0:
            print(f"{step}={losssum/log_interval},lr={opt.param_groups[0]['lr']}")
            print(trans_loss)
            print(rot_loss)
            print(cls_loss)
            with torch.no_grad():
                no = random.randint(0, 9)
                inputs = torch.tensor([[no]])
                print(f"show number {no}")
                pred_trans, repr6d, pred_mask = model(inputs)
                pred_rot_mat = model.get_rotation_matrix(repr6d.reshape(7, 6))
                print(pred_mask.shape)
                demo_rot(
                    pred_trans.reshape(7, 3),
                    pred_rot_mat.reshape(7, 3, 3),
                    pred_mask.reshape(7, 2),
                )


if __name__ == "__main__":
    train()
