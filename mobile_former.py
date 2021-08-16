import time
import torch
import torch.nn as nn
from mobile import Mobile, SeModule, hswish, MobileV3
from former import Former
from bridge import Mobile2Former, Former2Mobile


class MobileFormerBlock(nn.Module):
    def __init__(self, inp, exp, out, se=None, stride=1):
        super(MobileFormerBlock, self).__init__()
        if stride == 2:
            print('down sample')
        self.mobile = Mobile(3, inp, exp, out, se, stride)
        self.mobile2former = Mobile2Former(dim=192, heads=2, c=inp)
        self.former = Former(dim=192)
        self.former2mobile = Former2Mobile(dim=192, heads=2, c=out)

    def forward(self, inputs):
        x, z = inputs
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        return [x_out, z_out]


class Mobile_Former(nn.Module):
    def __init__(self, num_classes=1000, dyrelu=False):
        super(Mobile_Former, self).__init__()

        self.num_classes = num_classes
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, 6, 192)))

        # stem 3 224 224 -> 16 112 112
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish(),
        )
        self.bneck = MobileV3(3, 16, 32, 16, nn.ReLU(inplace=True), None, 1)
        # mobile-former
        self.block = nn.Sequential(
            MobileFormerBlock(16, 96, 24, SeModule(24), 2),
            MobileFormerBlock(24, 96, 24, None, 1),
            MobileFormerBlock(24, 144, 48, None, 2),
            MobileFormerBlock(48, 192, 48, SeModule(48), 1),
            MobileFormerBlock(48, 288, 96, SeModule(96), 2),
            MobileFormerBlock(96, 384, 96, SeModule(96), 1),
            MobileFormerBlock(96, 576, 128, SeModule(128), 1),
            MobileFormerBlock(128, 768, 128, SeModule(128), 1),
            MobileFormerBlock(128, 768, 192, SeModule(192), 2),
            MobileFormerBlock(192, 1152, 192, SeModule(192), 1),
            MobileFormerBlock(192, 1152, 192, SeModule(192), 1),
        )
        self.conv = nn.Conv2d(192, 1152, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(1152)
        self.avg = nn.AvgPool2d((7, 7))
        self.head = nn.Sequential(
            nn.Linear(1152 + 192, 1920),
            hswish(),
            nn.Linear(1920, self.num_classes)
        )

    def forward(self, x):
        b, _, _, _ = x.shape
        z = self.token.repeat(b, 1, 1)
        x = self.bneck(self.stem(x))
        x, z = self.block([x, z])
        x = self.avg(self.bn(self.conv(x))).view(b, -1)
        z = z[:, 0, :].view(b, -1)
        out = torch.cat((x, z), -1)
        return self.head(out)
        # return x, z


if __name__ == "__main__":
    model = Mobile_Former(num_classes=1000, dyrelu=True)
    inputs = torch.randn((3, 3, 224, 224))
    print(inputs.shape)
    # for i in range(100):
    #     t = time.time()
    #     output = model(inputs)
    #     print(time.time() - t)
    print("Total number of parameters in networks is {} M".format(sum(x.numel() for x in model.parameters()) / 1e6))
    output = model(inputs)
    print(output.shape)
