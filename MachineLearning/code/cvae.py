import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import sigpy as sp

l1_loss = torch.nn.L1Loss()
lpips_loss = lpips.LPIPS(net='vgg').to(torch.device("cuda"))
my_finite_difference = sp.to_pytorch_function(sp.linop.FiniteDifference(ishape=(64, 64)))

class Block(nn.Module):
    """Basic convolutional building block
    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int     
        number of output channels of the block
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.lrelu =  nn.LeakyReLU(0.1)

    def forward(self, x):
        """Performs a forward pass of the block
        x : torch.Tensor
            the input to the block
        torch.Tensor
            the output of the forward pass
        """
        # a block consists of two convolutional layers
        # with LeakyReLU activations
        # use batch normalisation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        return x


class Encoder(nn.Module):
    """The encoder part of the cVAE.
    Parameters
    ----------
    spatial_size : list[int]
        size of the input image, by default [64, 64]
    z_dim : int
        dimension of the latent space
    chs : tuple
        hold the number of input channels for each encoder block
    """
    def __init__(self, z_dim=256, chs=(2, 64, 128, 256), spatial_size=[64, 64]):
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        # max pooling
        self.pool = nn.MaxPool2d(2)
        # height and width of images at lowest resolution level
        _h, _w = (
            spatial_size[0] // (2**(len(chs) - 1)), 
            spatial_size[1] // (2**(len(chs) - 1))
        )
        # flattening
        self.out = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def forward(self, x, condition):
        """Performs the forward pass for all blocks in the encoder.
        Parameters
        ----------
        x : torch.Tensor
            input image to the encoder
        enc_condition : torch.Tensor
            encoded input mask to the encoder
        Returns
        -------
        list[torch.Tensor]    
            a tensor with the means and a tensor with the log variances of the
            latent distribution
        """
        # Concatenate input and condition in channel dimension
        x = torch.cat((x, condition), dim=1)
        for block in self.enc_blocks:
            x = block(x)          
            x = self.pool(x) 
        x = self.out(x)          
        return torch.chunk(x, 2, dim=1)  # 2 chunks, 1 each for mu and logvar
    

class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization
    Parameters
    ----------
    x_ch : int
        number of input channels
    condition_ch : int     
        number of condition channels
    conv_ch : int
        number of convolution channels for SPADE
    """
    def __init__(self, x_ch, condition_ch=1, conv_ch=128):
        super().__init__()
        self.norm = nn.BatchNorm2d(x_ch, affine=False)
        self.conv = nn.Sequential(
            nn.Conv2d(condition_ch, conv_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Convolutions for learnable gamma and beta tensors
        self.conv_gamma = nn.Conv2d(conv_ch, x_ch, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(conv_ch, x_ch, kernel_size=3, padding=1)

    def forward(self, x, condition):
        # Downsampling of condition to match shape of x
        condition = F.interpolate(condition, 
            size=(x.size(2), x.size(3)), 
            mode="nearest"
        ) 
        condition = self.conv(condition)
        return self.norm(x) * self.conv_gamma(condition) + self.conv_beta(condition)


class SPADEBlock(nn.Module):
    """Spatially-Adaptive Denormalization
    Parameters
    ----------
    in_ch : int
        number of input channels
    out_ch : int     
        number of output channels
    condition_ch : int
        number of condition channels
    """
    def __init__(self, in_ch, out_ch, condition_ch=1):
        super().__init__()
        # Main layer
        self.spade_1 = SPADE(in_ch, condition_ch)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.spade_2 = SPADE(out_ch, condition_ch)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # Residual layer
        self.spade_s = SPADE(in_ch, condition_ch)
        self.relu_s = nn.ReLU(inplace=True)
        self.conv_s = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    
    def forward(self, x, condition):
        y  = self.conv_1(self.relu_1(self.spade_1(x, condition)))
        y  = self.conv_2(self.relu_2(self.spade_2(y, condition)))
        y_ = self.conv_s(self.relu_s(self.spade_s(x, condition)))
        return y + y_
    

class Generator(nn.Module):
    """Generator of the cVAE
    Parameters
    ----------
    z_dim : int 
        dimension of latent space
    chs : tuple
        holds the number of channels for each block
    h : int, optional
        height of image at lowest resolution level, by default 8
    w : int, optional
        width of image at lowest resolution level, by default 8    
    """
    def __init__(self, z_dim=256, chs=(256, 128, 64, 32), h=8, w=8):
        super().__init__()
        self.z_dim = z_dim
        self.chs = chs
        self.h = h  
        self.w = w   

        self.projection = nn.Linear(
            self.z_dim, self.chs[0] * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x: torch.reshape(
            x, (-1, self.chs[0], self.h, self.w)
        )  # reshaping
        self.spade_blocks = nn.ModuleList([
            SPADEBlock(chs[i + 1], chs[i + 1]) for i in range(len(chs) - 1) 
        ])
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)]            
        ) # transposed convolution
        self.head = nn.Sequential(
            nn.Conv2d(self.chs[-1], 1, 1),
            nn.Tanh(),
        )  # output layer

    def forward(self, z, condition):
        """Performs the forward pass of generator
        Parameters
        ----------
        z : torch.Tensor
            input to the generator
        enc_condition : torch.Tensor
            encoded condition to the generator
        Returns
        -------
        x : torch.Tensor
        """
        x = self.projection(z)
        x = self.reshape(x)
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            x = self.spade_blocks[i](x, condition)
        return self.head(x)


class CVAE(nn.Module):
    """A representation of the VAE
    Parameters
    ----------
    enc_chs : tuple 
        holds the number of input channels of each block in the encoder
    dec_chs : tuple 
        holds the number of input channels of each block in the decoder
    """
    def __init__(self, z_dim=256):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.generator = Generator(z_dim)

    def forward(self, x, condition):
        """Performs a forwards pass of the VAE and returns the reconstruction
        and mean + logvar.
        Parameters
        ----------
        x : torch.Tensor
            the input to the encoder
        condition : torch.Tensor
            the condition to both models
        Returns
        -------
        torch.Tensor
            the reconstruction of the input image
        float
            the mean of the latent distribution
        float
            the log of the variance of the latent distribution
        """
        mu, logvar = self.encoder(x, condition)
        latent_z = sample_z(mu, logvar)
        output = self.generator(latent_z, condition)
        return output, mu, logvar


def get_noise(n_samples, z_dim, device="cpu"):
    """Creates noise vectors.
    Given the dimensions (n_samples, z_dim), creates a tensor of that shape filled with 
    random numbers from the normal distribution.
    Parameters
    ----------
    n_samples : int
        the number of samples to generate
    z_dim : int
        the dimension of the noise vector
    device : str
        the type of the device, by default "cpu"
    """
    return torch.randn(n_samples, z_dim, device=device)

def sample_z(mu, logvar):
    """Samples noise vector from a Gaussian distribution with reparameterization trick.
    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    """
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu

def kld_loss(mu, logvar):
    """Computes the KLD loss given parameters of the predicted 
    latent distribution.
    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    Returns
    -------
    float
        the kld loss
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def regularize_var(recon):
    """Function to calculate the variation in image, 
    which is used to regulate the loss"""
    variation_loss = 0
    batch_size = recon.shape[0]
    for i in range(batch_size):
        # Calculate variation
        variation = my_finite_difference.apply(recon[i][0])
        # Norm
        norm_variation = torch.norm(variation, p=2)**2
        # Calculate regularization loss
        variation_loss += torch.sqrt(norm_variation)

    return variation_loss

def vae_loss(inputs, recons, mu, logvar, beta=1.4):
    """Computes the VAE loss, sum of reconstruction and KLD loss
    Parameters
    ----------
    inputs : torch.Tensor
        the input images to the vae
    recons : torch.Tensor
        the predicted reconstructions from the vae
    mu : float
        the predicted mean of the latent distribution
    logvar : float
        the predicted log of the variance of the latent distribution
    Returns
    -------
    float
        sum of reconstruction and KLD loss
    """
    # Reconstruction loss: L1 loss
    recon_img_loss = l1_loss(inputs, recons)
    # Perceptual loss using LPIPS
    perceptual = lpips_loss(inputs, recons).mean()
    # Combined reconstruction loss (L1 + 0.5 * perceptual)
    recon_loss = recon_img_loss + 0.5 * perceptual
    # KLD loss
    kld = kld_loss(mu, logvar)
    # Variation regularization term
    variational_loss = regularize_var(recons)

    return recon_loss + beta*kld + 1e-4*variational_loss
