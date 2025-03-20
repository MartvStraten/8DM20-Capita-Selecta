import torch
import torch.nn as nn

class Block(nn.Module):
    """Class for the basic convolutional building block of the unet
    """
    def __init__(self, in_ch, out_ch):
        """Constructor.
        :param in_ch: number of input channels to the block
        :param out_ch: number of output channels of the block
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        """Returns the output of a forward pass of the block
        :param x: the input tensor
        :return: the output tensor of the block
        """
        # a block consists of two convolutional layers
        # with ReLU activations
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    """Class for the encoder part of the unet.
    """
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        """Constructor.
        :param chs: tuple giving the number of input channels of each block in the encoder
        """
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """Returns the list of the outputs of all the blocks in the encoder
        :param x: input image tensor
        """
        ftrs = []  # a list to store features
        for block in self.enc_blocks:
            x = block(x)
            # save features to concatenate to decoder blocks
            ftrs.append(x)
            x = self.pool(x)
        ftrs.append(x)
        return ftrs

class Decoder(nn.Module):
    """Class for the decoder part of the unet.
    """

    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        """Constructor.
        :param chs: tuple giving the number of input channels of each block in the decoder
        """
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i], 2, 2) for i in range(len(chs) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(2 * chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        """Returns the output of the decoder part of the unet
        :param x: input tensor to the decoder
        :param encoder_features: list of the encoder features to be concatenated to the corresponding level of the decoder
        """
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            # get the features from the corresponding level of the encoder
            enc_ftrs = encoder_features[i]
            # concatenate these features to x
            x = torch.cat([x, enc_ftrs], dim=1)
            # convolutional block
            x = self.dec_blocks[i](x)
        return x

class UNet(nn.Module):
    """Class for the unet
    """
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256),
        dec_chs=(256, 128, 64, 32),
        num_classes=1,
    ):
        """Constructor.
        :param enc_chs: tuple giving the number of input channels of each block in the encoder
        :param dec_chs: tuple giving the number of input channels of each block in the encoder
        :param num_classes: number of output classes of the segmentation
        """
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], num_classes, 1),
        )  # output layer

    def forward(self, x):
        """Returns the output of a forward pass of the unet
        :param x: the input tensor to the unet
        """
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        return out