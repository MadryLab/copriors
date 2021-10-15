import torch
import torch.nn as nn
import time

def CBR(nin, nout, kernel_size, stride=1, padding=0):
    return nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True),
        )

class MLPBlock(nn.Module):
    def __init__(self, nin, nout, kernel_size):
        super().__init__()
        self.conv1 = CBR(nin, nout, 1)
        self.conv2 = CBR(nout, nout, kernel_size, padding=int((kernel_size-1)/2))

    def forward(self, x):
        x = self.conv1(x)
        return x + self.conv2(x)

class ConvPool(nn.Module):
    def __init__(self, nin, nout, pool=2):
        super().__init__()
        self.conv = CBR(nin, nout, 1)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x):
        return self.pool(self.conv(x))

def blocklist(cls, nin, filters):
    layers = []
    curr_nin = nin
    for f in filters:
        layers.append(cls(curr_nin, f))
        curr_nin = f
    return nn.Sequential(*layers)

class BagnetCustom32(nn.Module):
    def __init__(self, nin, num_classes, input_size):
        # has receptive field size 17
        
        super().__init__()
        enc_filters = [256, 512, 512]
        self.pre_convs = nn.Sequential(
            MLPBlock(nin, enc_filters[0], 5), # rf: 5
            ConvPool(enc_filters[0], enc_filters[0]), # rf: 5
            MLPBlock(enc_filters[0], enc_filters[1], 3), # rf: 9
        )
        pool_filters = [1024, 1024]
        self.downsamples = blocklist(ConvPool, enc_filters[-1], pool_filters)

        num_downsamples = len(pool_filters) + 1
        input_ds = int(input_size/(2**num_downsamples))
        self.final_downsample = ConvPool(pool_filters[-1], pool_filters[-1], pool=input_ds) # pool the rest

        dec_filters = [512, 512, 256, 128]
        conv1_lambda = lambda nin, f: CBR(nin, f, 1)
        self.post_convs = blocklist(conv1_lambda, pool_filters[-1], dec_filters)
        self.fc = nn.Linear(dec_filters[-1], num_classes)

    def forward(self, x):
        x = self.pre_convs(x)
        x = self.downsamples(x)
        x = self.final_downsample(x)
        x = self.post_convs(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class BagnetCustom96Thin(nn.Module):
    def __init__(self, nin, num_classes, input_size):
        # has receptive field size 17
        
        super().__init__()
        enc_filters = [128, 256, 512]
        self.pre_convs = nn.Sequential(
            MLPBlock(nin, enc_filters[0], 5), # rf: 5
            ConvPool(enc_filters[0], enc_filters[0]), # rf: 5
            MLPBlock(enc_filters[0], enc_filters[1], 3), # rf: 9
            ConvPool(enc_filters[1], enc_filters[1]), # rf: 9
            MLPBlock(enc_filters[1], enc_filters[2], 3), # rf: 17
        )
        pool_filters = [1024, 1024]
        self.downsamples = blocklist(ConvPool, enc_filters[-1], pool_filters)

        num_downsamples = len(pool_filters) + 2
        input_ds = int(input_size/(2**num_downsamples))
        self.final_downsample = ConvPool(pool_filters[-1], pool_filters[-1], pool=input_ds) # pool the rest

        dec_filters = [512, 512, 256, 128]
        conv1_lambda = lambda nin, f: CBR(nin, f, 1)
        self.post_convs = blocklist(conv1_lambda, pool_filters[-1], dec_filters)
        self.fc = nn.Linear(dec_filters[-1], num_classes)

    def forward(self, x):
        x = self.pre_convs(x)
        x = self.downsamples(x)
        x = self.final_downsample(x)
        x = self.post_convs(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def bagnetcustom32(num_classes):
    return BagnetCustom32(nin=3, num_classes=num_classes, input_size=32)

def bagnetcustom96thin(num_classes):
    return BagnetCustom96Thin(nin=3, num_classes=num_classes, input_size=96)


