from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# import rigid
import torch
from draw3d import demo

save = "model.pt"


class MLP(nn.Module):
    def __init__(self, fin, *fo_list, act="none") -> None:
        super().__init__()
        layers = [] # nn.LayerNorm(fin)]
        fi = fin
        for i, fo in enumerate(fo_list):
            layers.append(nn.Linear(fi, fo))
            if i < len(fo_list) - 1 or act != "none":
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
        self.mlp = MLP(128, 64, act="")
        self.mlp2 = MLP(128, 64, act="")
        self.mlp3 = MLP(128, 64, act="")

        self.trans_net = MLP(64, 3 * 7)
        self.rot_net = MLP(64, 3 * 7)
        self.mask_net = MLP(64, 2 * 7)

    def forward(self, input):
        # input  # (B, L)
        x = self.emb(input)  # (B, L, D)
        # x = self.xy_projection(x)
        B, L, _ = x.shape
        # rigids = rigid.Rigid.identity(
        #     shape=(B, L, 8),
        #     dtype=x.dtype,
        #     device=x.device,
        #     requires_grad=self.training,
        #     fmt="quat"
        # )
        return self.trans_net(self.mlp(x)), self.rot_net(self.mlp2(x)), self.mask_net(self.mlp3(x))


def train():
    train_iters = 20000
    log_interval = 4000
    lr = 1e-3
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    import dataset3d as ds
    import random

    cls_loss_fn = torch.nn.CrossEntropyLoss()
    sumloss = 0
    for step in range(train_iters+1):
        inputs = []
        targets = []
        masks = []
        B = 16
        for b in range(B):
            no = random.randint(0, 9)
            inputs.append([no])
            targets.append(ds.NUMBERS[no])
            masks.append(ds.MASKS[no])
        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)
        masks = torch.tensor(masks)

        pred_trans, pred_rot, pred_mask = model(inputs)

        trans_loss = (pred_trans.reshape(B, 7, 3) - targets[:, :, :3])[:, :, :2].sum(
            axis=-1
        )
        trans_loss = ((masks * trans_loss).square()).sum()

        rot_loss = (pred_rot.reshape(B, 7, 3) - targets[:, :, 3:])[:, :, -1:].sum(
            axis=-1
        )
        rot_loss = ((masks * rot_loss) ** 2).sum()

        # print(mask.shape)
        cls_loss = cls_loss_fn(pred_mask.reshape(B * 7, 2), masks.reshape(B * 7).to(torch.int64))

        # print(trans_loss)
        # print(rot_loss)
        # print(cls_loss)

        loss = trans_loss + rot_loss + cls_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        sumloss += loss.detach().item()
        # for g in opt.param_groups:
        #     if step < 100:
        #         g["lr"] = lr * min(1, step / 100)
        #     elif step < train_iters * 0.9:
        #         g["lr"] = lr
        #     else:
        #         g["lr"] = lr * min(1, (train_iters - step) / (train_iters * 0.1))

        if step % log_interval == 0:
            print(f"{step}={sumloss/log_interval},lr={opt.param_groups[0]['lr']}")
            with torch.no_grad():
                no = random.randint(0, 9)
                inputs = torch.tensor([[no]])
                print(f"show number {no}")
                pred_trans, pred_rot, pred_mask = model(inputs)
                print(pred_mask.shape)
                demo(
                    pred_trans.reshape(7, 3),
                    pred_rot.reshape(7, 3),
                    pred_mask.reshape(7, 2),
                )


if __name__ == "__main__":
    train()
