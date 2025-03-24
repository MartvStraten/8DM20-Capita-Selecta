import torch
import torch.nn as nn

l1_loss = torch.nn.L1Loss()

class SPADE(nn.Module):
    def __init__(self, feature_dim, condition_dim=256):
        super().__init__()
        self.scale = nn.Linear(condition_dim, feature_dim)
        self.shift = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, condition):
        gamma = self.scale(condition).unsqueeze(-1).unsqueeze(-1)
        beta = self.shift(condition).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


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
        self.bn1 = nn.BatchNorm2d()
        self.spade1 = SPADE(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d()
        self.spade2 = SPADE(out_ch)
        self.lrelu =  nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

    def forward(self, x, condition=None, leaky=True):
        """Performs a forward pass of the block
        x : torch.Tensor
            the input to the block
        torch.Tensor
            the output of the forward pass
        """
        # If there is no condition given to the block
        if condition is None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            if leaky:
                x = self.lrelu(x)
            else:
                x = self.relu(x)
        # If there is a condition given to the block
        if condition is not None:
            x = self.conv1(x)
            x = self.spade1(x, condition)
            x = self.conv2(x)
            x = self.spade2(x, condition)
            if leaky:
                x = self.lrelu(x)
            else:
                x = self.relu(x)
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
        condition : torch.Tensor
            input mask to the encoder
        Returns
        -------
        list[torch.Tensor]    
            a tensor with the means and a tensor with the log variances of the
            latent distribution
        """
        x = torch.cat((x, condition), 1) # Concatenate input and condition
        for block in self.enc_blocks:
            x = block(x)          
            x = self.pool(x) 
        x = self.out(x)          
        return torch.chunk(x, 2, dim=1)  # 2 chunks, 1 each for mu and logvar

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
    def __init__(self, z_dim=256, chs_enc=(1, 64, 128, 256), chs_dec=(256, 128, 64, 32), h=8, w=8):
        super().__init__()
        self.chs_enc = chs_enc
        self.chs_dec = chs_dec
        self.h = h  
        self.w = w  
        self.z_dim = z_dim 

        # model to encode images to a vector
        self.image_encoder = nn.ModuleList(
            [Block(chs_enc[i], chs_enc[i + 1]) for i in range(len(chs_enc) - 1)]
        )
        # max pooling
        self.pool = nn.MaxPool2d(2)
        # flattening
        self.out = nn.Sequential(nn.Flatten(1), nn.Linear(chs_enc[-1] * self.h * self.w, self.z_dim))

        self.projection = nn.Linear(
            2 * self.z_dim, self.chs_dec[0] * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x, bs: torch.reshape(
            x, (bs, self.chs_dec[0], self.h, self.w)
        )  # reshaping
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs_dec[i], chs_dec[i + 1], 2, 2) for i in range(len(chs_dec) - 1)]            
        ) # transposed convolution
        self.dec_blocks = nn.ModuleList(
            [Block(chs_dec[i + 1], chs_dec[i + 1]) for i in range(len(chs_dec) - 1)]           
        ) # conv blocks
        self.head = nn.Sequential(
            nn.Conv2d(self.chs_dec[-1], 1, 1),
            nn.Tanh(),
        )  # output layer

    def forward(self, z, condition):
        """Performs the forward pass of generator
        Parameters
        ----------
        z : torch.Tensor
            input to the generator
        condition : torch.Tensor
            condition to the generator
        Returns
        -------
        x : torch.Tensor
        """
        # batch size
        bs = condition.shape[0]

        # encode condition image
        for block in self.image_encoder:
            condition = block(condition)          
            condition = self.pool(condition) 
        condition = self.out(condition) # [batch_size, z_dim]  

        # concatenate noise vector and condition
        x = torch.cat((z, condition), -1)
        x = self.projection(x)
        x = self.reshape(x, bs)
        for i in range(len(self.chs_dec) - 1):
            x = self.upconvs[i](x)
            x = self.dec_blocks[i](x)
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

def vae_loss(inputs, recons, mu, logvar, beta=1):
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
    return l1_loss(inputs, recons) + beta*kld_loss(mu, logvar)
