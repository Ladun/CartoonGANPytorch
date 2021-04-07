import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels)
        )
    def forward(self, x):

        return x + self.model(x)


class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()
        
        self.config = config

        self.flat = nn.Sequential(
            nn.Conv2d(3, self.config.gen_n_filters, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(self.config.gen_n_filters),
            nn.LeakyReLU(inplace=True)
        )

        self.down_sampling = nn.Sequential(
            nn.Conv2d(self.config.gen_n_filters, self.config.gen_n_filters * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.config.gen_n_filters * 2, self.config.gen_n_filters * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.config.gen_n_filters * 2),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(self.config.gen_n_filters * 2, self.config.gen_n_filters * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.config.gen_n_filters * 4, self.config.gen_n_filters * 4,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.config.gen_n_filters * 4),
            nn.LeakyReLU(inplace=True),
        )

        res_blocks = []
        for _ in range(8):
            res_blocks.append(ResidualBlock(self.config.gen_n_filters * 4))
        self.residual_blocks = nn.Sequential(*res_blocks)

        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(self.config.gen_n_filters * 4, self.config.gen_n_filters * 2,
                               kernel_size=3,stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.config.gen_n_filters * 2, self.config.gen_n_filters * 2,
                               kernel_size=3,stride=1, padding=1),
            nn.InstanceNorm2d(self.config.gen_n_filters * 2),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(self.config.gen_n_filters * 2, self.config.gen_n_filters,
                               kernel_size=3,stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.config.gen_n_filters, self.config.gen_n_filters,
                               kernel_size=3,stride=1, padding=1),
            nn.InstanceNorm2d(self.config.gen_n_filters),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.config.gen_n_filters, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.flat(x)
        x = self.down_sampling(x)
        x = self.residual_blocks(x)
        x = self.up_sampling(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.config = config

        model = [
            nn.Conv2d(3, self.config.disc_n_filters, kernel_size=3,stride=1, padding=1),
            nn.LeakyReLU(self.config.leakyrelu_negative_slope, inplace=True),

            nn.Conv2d(self.config.disc_n_filters, self.config.disc_n_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(self.config.leakyrelu_negative_slope, inplace=True),
            nn.Conv2d(self.config.disc_n_filters * 2, self.config.disc_n_filters * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.config.disc_n_filters * 4),
            nn.LeakyReLU(self.config.leakyrelu_negative_slope, inplace=True),
            
            nn.Conv2d(self.config.disc_n_filters * 4, self.config.disc_n_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(self.config.leakyrelu_negative_slope, inplace=True),
            nn.Conv2d(self.config.disc_n_filters * 4, self.config.disc_n_filters * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.config.disc_n_filters * 8),
            nn.LeakyReLU(self.config.leakyrelu_negative_slope, inplace=True),
            
            nn.Conv2d(self.config.disc_n_filters * 8, self.config.disc_n_filters * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.config.disc_n_filters * 8),
            nn.LeakyReLU(self.config.leakyrelu_negative_slope, inplace=True),

            nn.Conv2d(self.config.disc_n_filters * 8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        x = self.model(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()

        assert config.feature_network in ['vgg', 'resnet-101']


        if config.feature_network == 'resnet-101':
            model = torchvision.models.resnet101(pretrained=True)
            layers = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2]
            self.model = nn.Sequential(*layers)
        
        elif config.feature_network == 'vgg':
            vgg = torchvision.models.vgg19_bn(pretrained=True)
            self.model = vgg.features[:37]

        # FeatureExtractor should not be trained
        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False


    def forward(self, x):
        x = self.model(x)
        return x
