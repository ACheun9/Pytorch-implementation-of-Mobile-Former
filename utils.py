import torch
import torch.nn as nn


class MyDyRelu(nn.Module):
    def __init__(self, k):
        super(MyDyRelu, self).__init__()
        self.k = k

    def forward(self, inputs):
        x, relu_coefs = inputs
        # BxCxHxW -> HxWxBxCx1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        # h w b c 1 -> _ _ b c k
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        # HxWxBxCxk -> BxCxHxW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result

