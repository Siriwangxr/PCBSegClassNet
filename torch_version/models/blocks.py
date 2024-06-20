import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="conv", kernel_size=3, stride=1, padding=1, relu=True, upsampling=False, up_sample_size=2, skip_layer=None):
        super(ConvBlock, self).__init__()

        if conv_type == "ds":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels)
        elif conv_type == "conv":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            raise ValueError("Wrong choice of convolution type.")
        self.relu_flag = relu

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

        self.upsampling = upsampling
        if upsampling:
            self.upsample = nn.Upsample(scale_factor=up_sample_size, mode='bilinear', align_corners=True)
            self.skip_conv = nn.Conv2d(in_channels // 8, out_channels, kernel_size=1, stride=1, padding=0)


            if conv_type == "ds":
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=out_channels * 2),
                    nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
                )
            elif conv_type == "conv":
                self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size, 1, padding)
            else:
                raise ValueError("Wrong choice of convolution type.")

            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x, skip_layer=None):
        out = self.conv(x)
        out = self.bn(out)

        if self.relu_flag:
            out = self.relu(out)

        if self.upsampling:
            out = self.upsample(out)
            if skip_layer is not None:
                skip_layer = self.skip_conv(skip_layer)
                out = torch.cat([out, skip_layer], dim=1)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)

        return out

class ResidualBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, temp=3, relu=False):
        super(ResidualBottleneck, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels * temp, "conv", 1, 1, 0)
        self.dconv = nn.Conv2d(in_channels * temp, in_channels * temp, kernel_size, stride, padding=kernel_size // 2, groups=in_channels * temp)
        self.bn = nn.BatchNorm2d(in_channels * temp)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvBlock(in_channels * temp, out_channels, "conv", 1, 1, 0, relu=False)
        self.relu_out = nn.ReLU(inplace=True) if relu else nn.Identity()
        # self.shortcut = nn.Sequential()
        # if in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )
        self.relu_flag = relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.dconv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.relu_flag:
            # out += self.shortcut(x)
            out += x

        return out

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, temp=6, loop=3):
        super(BottleneckBlock, self).__init__()

        self.bottlenecks = nn.ModuleList()
        for i in range(loop):
            if i == 0:
                self.bottlenecks.append(ResidualBottleneck(in_channels, out_channels, kernel_size, stride, temp, relu=False))
            else:
                self.bottlenecks.append(ResidualBottleneck(out_channels, out_channels, kernel_size, 1, temp, relu=True))

    def forward(self, x):
        for bottleneck in self.bottlenecks:
            x = bottleneck(x)
        return x

class PyramidPoolingBlock(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(PyramidPoolingBlock, self).__init__()

        width = 16
        height = 16
        self.pools = nn.ModuleList()
        for bin_size in bin_sizes:
            self.pools.append(nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=(width // bin_size),
                    stride=(width // bin_size)
                ),
                nn.Conv2d(in_channels, 64, 3, 2, 1),
                nn.Upsample(size=(width, height), mode='bilinear', align_corners=False)
            ))

    def forward(self, x):
        out = [x]
        for pool in self.pools:
            out.append(pool(x))
        out = torch.cat(out, dim=1)
        return out

class TemBlock(nn.Module):
    def __init__(self, in_channels):
        super(TemBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 256, 3, 1, 1)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(256, 256, 1, 1, 0)

    def forward(self, x):
        conv1 = self.conv1(x)
        gap1 = self.gap1(conv1)
        conv2 = self.conv2(gap1)
        conv3 = self.conv3(conv2)
        mult1 = conv1 * conv3
        # B
        add1 = conv1 + mult1

        gap2 = self.gap2(add1)
        cos_sim1 = F.cosine_similarity(gap2, add1, dim=1)
        reshape1 = cos_sim1.view(cos_sim1.shape[0], -1)  # C

        mat_mul1 = torch.matmul(reshape1.transpose(1, 0), reshape1)

        conv4 = self.conv4(add1)
        permute1 = conv4.permute(0, 2, 3, 1)
        reshape2 = permute1.reshape(permute1.shape[0], -1, permute1.shape[1] * permute1.shape[2])
        mat_mul2 = torch.matmul(reshape2, mat_mul1)
        reshape3 = mat_mul2.view(mat_mul2.shape[0], -1, conv4.shape[2], conv4.shape[3])
        permute2 = reshape3.permute(0, 1, 2, 3)

        return permute2

class LearningModule(nn.Module):
    def __init__(self):
        super(LearningModule, self).__init__()

        self.layer1 = ConvBlock(3, 16, "conv", 3, 2, 1, relu=True)
        self.layer2 = ConvBlock(16, 32, "ds", 3, 2, 1, relu=True)
        self.layer3 = ConvBlock(32, 48, "conv", 3, 2, 1, relu=True)

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return layer1, layer2, layer3

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.layer1 = BottleneckBlock(48, 48, 3, 2, 6, 3)
        self.layer2 = BottleneckBlock(48, 64, 3, 2, 6, 3)
        self.layer3 = BottleneckBlock(64, 96, 3, 1, 6, 3)
        self.layer4 = PyramidPoolingBlock(96, [2, 4, 6, 8])

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4

class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()

        self.conv1 = ConvBlock(48, 96, "conv", 1, 1, 0, relu=True)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.dconv = nn.Conv2d(352, 352, 3, 1, 1, groups=352)
        self.bn = nn.BatchNorm2d(352)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(352, 96, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(96)
    def forward(self, learning_layer, fe_layer):
        fusion_layer1 = self.conv1(learning_layer)
        fusion_layer2 = self.upsample(fe_layer)
        fusion_layer2 = self.dconv(fusion_layer2)
        fusion_layer2 = self.bn(fusion_layer2)
        fusion_layer2 = self.relu(fusion_layer2)
        fusion_layer2 = self.conv2(fusion_layer2)

        fusion_layer = fusion_layer1 + fusion_layer2
        fusion_layer = self.bn1(fusion_layer)
        fusion_layer = self.relu(fusion_layer)

        return fusion_layer

class PCBEncoder(nn.Module):
    def __init__(self):
        super(PCBEncoder, self).__init__()

        self.learning_module = LearningModule()
        self.feature_extractor = FeatureExtractor()
        self.fusion_module = FusionModule()

    def forward(self, x):
        learning_layer1, learning_layer2, learning_layer3 = self.learning_module(x)
        fe_layer = self.feature_extractor(learning_layer3)
        fusion_layer = self.fusion_module(learning_layer3, fe_layer)
        return fusion_layer, learning_layer1, learning_layer2

class PCBDecoder(nn.Module):
    def __init__(self, num_classes):
        super(PCBDecoder, self).__init__()

        self.tem = TemBlock(96)
        self.classifier1 = ConvBlock(256, 128, "conv", 3, 1, 1, relu=True, upsampling=True, up_sample_size=2)
        self.classifier2 = ConvBlock(128, 128, "ds", 3, 1, 1, relu=True, upsampling=True, up_sample_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.classifier3 = nn.Sequential(
            nn.Conv2d(128, num_classes, 1, 1, 0),
            nn.BatchNorm2d(num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, encoder_output, learning_layer1, learning_layer2):
        out = self.tem(encoder_output)
        classifier1 = self.classifier1(out, learning_layer2)
        classifier2 = self.classifier2(classifier1, learning_layer1)
        classifier2 = self.upsample(classifier2)
        classifier3 = self.classifier3(classifier2)
        return classifier3

class PCBModel(nn.Module):
    def __init__(self, num_classes):
        super(PCBModel, self).__init__()

        self.encoder = PCBEncoder()
        self.decoder = PCBDecoder(num_classes)

    def forward(self, x):
        fusion_layer, learning_layer1, learning_layer2 = self.encoder(x)
        output = self.decoder(fusion_layer, learning_layer1, learning_layer2)
        return output