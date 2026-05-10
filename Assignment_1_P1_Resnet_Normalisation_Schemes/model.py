import torch
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset


class in100_dataset(Dataset):
    def __init__(self, root_dir, classes, class_to_idx, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_dir, cls_name)

            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)

                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_layer):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = norm_layer(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(out_channels),
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.norm1(self.conv1(x))
        out = self.relu(out)
        out = self.norm2(self.conv2(out))

        out += identity

        return self.relu(out)


class resnet_layer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_layer):
        super().__init__()

        self.block1 = residual_block(in_channels, out_channels, stride, norm_layer)
        self.block2 = residual_block(out_channels, out_channels, 1, norm_layer)

    def forward(self, x):
        return self.block2(self.block1(x))


class resnet18(nn.Module):
    def __init__(self, num_classes, norm_layer):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = resnet_layer(64, 64, 1, norm_layer)
        self.layer2 = resnet_layer(64, 128, 2, norm_layer)
        self.layer3 = resnet_layer(128, 256, 2, norm_layer)
        self.layer4 = resnet_layer(256, 512, 2, norm_layer)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif (
                hasattr(m, "weight")
                and m.weight is not None
                and len(m.weight.shape) == 1
            ):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class no_norm(nn.Module):
    def __init__(self, num_features):
        super().__init__()

    def forward(self, x):
        return x


class batch_norm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.momentum = 0.1

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        with torch.no_grad():
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)

            self.running_mean.mul_(1 - self.momentum).add_(
                self.momentum * mean.detach()
            )
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())

        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean[None, :, None, None]) / torch.sqrt(
            var[None, :, None, None] + 1e-5
        )
        x_hat = self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]

        return x_hat


class instance_norm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + 1e-5)
        x_hat = self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]

        return x_hat


class batch_instance_norm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = batch_norm(num_features)
        self.ins_norm = instance_norm(num_features)
        self.rho = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        out_bn = self.bn(x)
        out_in = self.ins_norm(x)

        rho = torch.clamp(self.rho, 0, 1)
        rho = rho[None, :, None, None]

        return rho * out_bn + (1 - rho) * out_in


class layer_norm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), unbiased=False, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + 1e-5)
        x_hat = self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]

        return x_hat


class group_norm(nn.Module):
    def __init__(self, num_features, num_groups=32):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups

        assert num_features % num_groups == 0

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        N, C, H, W = x.shape
        G = self.num_groups

        assert C % G == 0

        x = x.view(N, G, C // G, H, W)

        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

        x = (x - mean) / torch.sqrt(var + 1e-5)
        x = x.view(N, C, H, W)
        x = self.gamma[None, :, None, None] * x + self.beta[None, :, None, None]

        return x
