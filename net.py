from torch import nn
import numpy as np
# import rigid
import torch 

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

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        print(x.shape)
        return self.layers(x)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.emb = nn.Embedding(10, 16)
        self.mlp = MLP(16, 64, 32, 16)

        self.act_fn = nn.ReLU()
        self.trans_net = MLP(16, 3*8)
        self.rot_net = MLP(16, 3*8)
        self.mask_net = MLP(16, 2*8)

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
        return self.trans_net(x), self.rot_net(x), self.mask_net(x)


def train():
    train_iters = 1000
    log_interval = 10
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    import dataset as ds
    import random

    cls_loss = torch.nn.CrossEntropyLoss()
    for step in range(train_iters):
        no = random.randint(0, 9)
        input = torch.tensor([[no]])
        target = torch.tensor([ds.NUMBERS[no]])
        mask = torch.tensor([ds.MASKS[no]])

        pred_trans, pred_rot, pred_mask = model(input)

        trans_loss = (pred_trans.reshape(1, 8, 3) - target[:, :, :3]).sum(axis=-1)
        trans_loss = ((mask * trans_loss) ** 2).sum()

        rot_loss = (pred_rot.reshape(1, 8, 3) - target[:, :, 3:]).sum(axis=-1)
        rot_loss = ((mask * rot_loss) ** 2).sum()

        cls_loss(pred_mask.reshape(1,8,2), mask)

        loss = trans_loss + rot_loss + cls_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_interval:
            print(f"{step}={loss}")

if __name__ == "__main__":
    train()


