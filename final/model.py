import math
import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out = self.body(x)
        return out
    
class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.body(x)
        return out

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel , channel , 3, 1, 3, 3)
        self.c2 = BasicBlock(channel , channel , 3, 1, 5, 5)
        self.c3 = BasicBlock(channel , channel , 3, 1, 7, 7)
        self.c4 = BasicBlockSig(channel*3, channel , 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y


class SuperResolution(nn.Module):
    def __init__(self, scale_factor=3, num_channels=3, d=56, s=12, m=16):
        super(SuperResolution, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d),
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )

        self.mid_part1 = []
        for _ in range(m//2):
            self.mid_part1.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part1 = nn.Sequential(*self.mid_part1)

        self.mid_part2 = []
        for _ in range(m//2):
            self.mid_part2.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part2 = nn.Sequential(*self.mid_part2)

        self.ca = CALayer(s)

        # upscaling
        self.upscale3x = nn.Sequential(
            nn.Conv2d(s, s * 9, kernel_size=3, stride=1, padding=1), 
            nn.PixelShuffle(3),
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )

        self.last_part = nn.Conv2d(in_channels=d, out_channels=num_channels, kernel_size=3, stride=1, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part1:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part2:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.ca.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.upscale3x:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        residual = x
        x = self.mid_part1(x) + x
        x = self.mid_part2(x) + x
        x = self.ca(x)
        x = x + residual
        x = self.upscale3x(x)
        x = self.last_part(x)
        return x