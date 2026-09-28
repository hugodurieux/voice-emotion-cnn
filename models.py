"""The two CNNs: a small one trained from scratch, and a frozen ImageNet ResNet-18."""
import torch
import torch.nn as nn
import torchvision


class ScratchCNN(nn.Module):
    """Four conv blocks (conv3x3 + BN + ReLU + 2x2 max-pool), 16->32->64->128 channels,
    global average pooling, dropout, one fully connected layer. Kept small for a laptop CPU."""

    def __init__(self, n_classes=7):
        super().__init__()
        layers, c_in = [], 1
        for c_out in [16, 32, 64, 128]:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ResNetEmbedder(nn.Module):
    """ResNet-18 pretrained on ImageNet, frozen, with its classifier removed.
    The spectrogram is rescaled to [0, 1], copied into three channels and normalised like an RGB image."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Identity()
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):                      # x: (B, 1, 64, 301)
        lo = x.amin(dim=(2, 3), keepdim=True)
        hi = x.amax(dim=(2, 3), keepdim=True)
        x = (x - lo) / (hi - lo + 1e-6)
        x = x.repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return self.net(x)                     # (B, 512)


def transfer_head(n_classes=7):
    """The only trainable part of the transfer-learning model."""
    return nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_classes))
