import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
try:
    from types_ import *
except:
    from typing import List, Callable, Union, Any, TypeVar, Tuple
    Tensor = TypeVar('torch.tensor')


def con2d_layers(h_in, w_in, padding, kernel_size, stride, dilation=(1, 1)):
    """

    :param h_in: Height of input --> int
    :param w_in: Width of input --> int
    :param padding: --> tuple (int, int)
    :param kernel_size: --> tuple (int, int)
    :param stride: --> tuple (int, int)
    :param dilation: --> tuple (int, int). default: (1, 1)
    :return: (output height, output width) --> tuple (int, int)
    """
    h_out = 1 + (h_in + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) - 1)/stride[0]
    w_out = 1 + (w_in + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]
    return int(h_out), int(w_out)


def contrans2d_layers(h_in, w_in, padding, kernel_size, stride, output_padding=(0, 0), dilation=(1, 1)):
    """

    :param h_in: Height of input --> int
    :param w_in: Width of input --> int
    :param padding: --> tuple (int, int)
    :param kernel_size: --> tuple (int, int)
    :param stride: --> tuple (int, int)
    :param output_padding: --> tuple (int, int). default: (0, 0)
    :param dilation: --> tuple (int, int). default: (1, 1)
    :return: (output height, output width) --> tuple (int, int)
    """
    h_out = (h_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    w_out = (w_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
    return int(h_out), int(w_out)


class SAEC(nn.Module):
    num_iter = 0

    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 6,
                 hidden_dims: List = None,
                 beta: int = 0.1,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type: str = 'H',
                 loss_weight: float = 3.,
                 l1: int = None,
                 l2: int = None,
                 activation=nn.Tanh(),
                 activation_latent=nn.Identity(),
                 **kwargs) -> None:
        super(SAEC, self).__init__()

        self.loss_weight = loss_weight
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter
        self.activation_latent = activation_latent  # nn.LeakyReLU()
        self.activation = activation
        modules = []

        if hidden_dims is None:
            hidden_dims = [2, 8]

        ind = 0
        h_in, w_in = 100, 3
        for h_dim in hidden_dims:
            h_out, w_out = con2d_layers(h_in, w_in, (2, 1-ind), (6, 3), (2, 1))
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=(6, 3),
                              stride=(2, 1),
                              padding=(2, 1 - ind)
                              ),
                    self.activation)
            )
            in_channels = h_dim
            ind += 1
            h_in, w_in = h_out, w_out

        self.encoder = nn.Sequential(*modules)

        # Build the middle linear layers:

        if l1 is None or l2 is None:
            self.l1 = hidden_dims[-1] * h_out * w_out
            self.l2 = hidden_dims[-1] * h_out * w_out
        else:
            self.l1 = l1
            self.l2 = l2

        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1] * h_out * w_out, self.l1),
                self.activation,
                nn.Linear(self.l1, hidden_dims[0]*h_out*w_out),
                self.activation,
                nn.Linear(hidden_dims[0]*h_out*w_out, latent_dim),
                self.activation_latent,
                nn.Linear(latent_dim, latent_dim)
            )
        )
        self.fc_z = nn.Sequential(*modules)

        # Build Linear Layers Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                self.activation_latent,
                nn.Linear(latent_dim, hidden_dims[0]*h_out*w_out),
                self.activation,
                nn.Linear(hidden_dims[0]*h_out*w_out, hidden_dims[-1] * h_out * w_out),
                self.activation,
                nn.Linear(hidden_dims[-1] * h_out * w_out, hidden_dims[-1] * h_out * w_out),
                self.activation
            )
        )
        self.decoder_input = nn.Sequential(*modules)

        # Build Decoder
        modules = []
        hidden_dims.reverse()
        ind = 0
        h_in, w_in = h_out, w_out
        for index, i in enumerate(hidden_dims):
            try:
                out = hidden_dims[index + 1]
            except:
                out = 1
            h_out, w_out = contrans2d_layers(h_in, w_in, (0, 0), (2, 2), (2, 1))
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(i,
                                       out,
                                       kernel_size=(2, 2),
                                       stride=(2, 1),
                                       padding=(0, 0),
                                       dilation=(1, 1)
                                       ),
                    self.activation)
            )
            ind += 1
            h_in, w_in = h_out, w_out
        self.decoder = nn.Sequential(*modules)

    def encode(self, input_: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input_: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) latent tensor
        """
        result = self.encoder(input_)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        return z

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 8, 25, 1)
        result = self.decoder(result)
        return result

    def forward(self, input_: Tensor, **kwargs) -> Tensor:
        z = self.encode(input_)
        return [self.decode(z), input_, z]

    def distance(self, p_pred, p):
        p_pred, p = p_pred.reshape(p_pred.shape[1], p_pred.shape[2]), p.reshape(p.shape[1], p.shape[2])
        d = torch.tensor([torch.sqrt(torch.sum((p[i, :] - p_pred[i, :])**2)) for i in range(p_pred.shape[0])])
        return torch.mean(d)

    def loss_latent_constraints(self,
                                *args,
                                **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input_ = args[1]
        z = args[2]
        force = kwargs['a']
        recons_loss = torch.sqrt(F.mse_loss(recons, input_))

        ### Classic MSE Loss force and torques:
        a_loss_torque = torch.sqrt(F.mse_loss(z[:, :3], force[:, :3]))  # , reduction="sum"
        a_loss_force = torch.sqrt(F.mse_loss(z[:, 3:], force[:, 3:]))  # , reduction="sum"

        ### Using decoder to create same scale of loss:
        from_latent = self.decode(force)
        latent_loss = torch.sqrt(F.mse_loss(from_latent, input_))
        loss = recons_loss + self.loss_weight*latent_loss + self.loss_weight * a_loss_torque + \
               self.loss_weight * a_loss_force
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'a torque': a_loss_torque, 'a force': a_loss_force,
                'latent loss': latent_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class SAEL(nn.Module):

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 scaler_layers: int = 2,
                 latent_dim: int = 6,
                 hidden_dims: List = None,
                 beta: int = 1,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type: str = 'H',
                 loss_weight: float = 3.,
                 activation=nn.Tanh(),
                 activation_latent=nn.Identity(),
                 **kwargs) -> None:
        super(SAEL, self).__init__()
        self.scaler_layers = scaler_layers
        self.loss_weight = loss_weight
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter
        self.activation_latent = activation_latent  # nn.LeakyReLU()
        self.activation = activation

        # Build Encoder
        ind = 0
        h_in, w_in = 100, 3
        layers = []
        modules = []
        while h_in * w_in / self.latent_dim > 2:
            modules.append(
                nn.Sequential(
                    nn.Linear(h_in * w_in, int(h_in * w_in / self.scaler_layers)),
                    self.activation,
                )
            )
            h_in, w_in = int(h_in * w_in / self.scaler_layers), 1
            layers.append(int(h_in * w_in / self.scaler_layers))
        modules.append(nn.Linear(layers[-2], latent_dim))

        h_in, w_in = self.latent_dim, 1
        self.encoder = nn.Sequential(*modules)

        # Build Linear Layers Decoder
        layers.reverse()
        modules = []
        for i in layers[1:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(h_in*w_in, i),
                    self.activation
                )
            )
            h_in, w_in = i, 1
        modules.append(nn.Linear(h_in*w_in, 100 * 3))
        self.decoder = nn.Sequential(*modules)

    def encode(self, input_: Tensor) -> List[Tensor]:
        """
        Encodes the input_ by passing through the encoder network
        and returns the latent codes.
        :param input_: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input_)
        return result

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        return result

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = torch.flatten(x, start_dim=1)
        z = self.encode(x)
        return [self.decode(z), x, z]

    def loss_latent_constraints(self,
                                *args,
                                **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        force = kwargs['a']
        recons_loss = torch.sqrt(F.mse_loss(recons, input))


        ### Classic MSE Loss force and torques:
        a_loss_torque = torch.sqrt(F.mse_loss(mu[:, :3], force[:, :3]))  # , reduction="sum"
        a_loss_force = torch.sqrt(F.mse_loss(mu[:, 3:], force[:, 3:]))  # , reduction="sum"
        # loss = recons_loss + self.loss_weight * a_loss_torque + self.loss_weight * 1.2 * a_loss_force

        ### Using decoder to create same scale of loss:
        from_latent = self.decode(force)
        latent_loss = torch.sqrt(F.mse_loss(from_latent, input))

        loss = recons_loss + self.loss_weight*latent_loss + self.loss_weight * a_loss_torque + \
               self.loss_weight * a_loss_force

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'a torque': a_loss_torque, 'a force': a_loss_force,
                'latent loss': latent_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class VAE(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 6,
                 hidden_dims: List = None,
                 beta: int = 0.01,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type: str = 'H',
                 loss_weight: float = 3.,
                 l1: int = None,
                 l2: int = None,
                 activation=nn.Tanh(),
                 activation_latent=nn.Identity(),
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.loss_weight = loss_weight
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter
        self.activation_latent = activation_latent  # nn.LeakyReLU()
        self.activation = activation
        modules = []
        if hidden_dims is None:
            hidden_dims = [2, 8]

        ind = 0
        h_in, w_in = 100, 3
        for h_dim in hidden_dims:
            h_out, w_out = con2d_layers(h_in, w_in, (2, 1-ind), (6, 3), (2, 1))
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=(6, 3),
                              stride=(2, 1),
                              padding=(2, 1 - ind)
                              ),
                    self.activation)
            )
            in_channels = h_dim
            ind += 1
            h_in, w_in = h_out, w_out

        self.encoder = nn.Sequential(*modules)

        # Build the middle linear layers:

        if l1 is None or l2 is None:
            self.l1 = hidden_dims[-1] * h_out * w_out
            self.l2 = hidden_dims[-1] * h_out * w_out
        else:
            self.l1 = l1
            self.l2 = l2

        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1] * h_out * w_out, self.l1),
                self.activation,
                nn.Linear(self.l1, hidden_dims[0]*h_out*w_out),
                self.activation,
                nn.Linear(hidden_dims[0]*h_out*w_out, latent_dim),
                self.activation_latent,
                nn.Linear(latent_dim, latent_dim)
            )
        )
        self.fc_mu = nn.Sequential(*modules)
        self.fc_mu1 = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        # Build Linear Layers Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                self.activation_latent,
                nn.Linear(latent_dim, hidden_dims[0]*h_out*w_out),
                self.activation,
                nn.Linear(hidden_dims[0]*h_out*w_out, hidden_dims[-1] * h_out * w_out),
                self.activation,
                nn.Linear(hidden_dims[-1] * h_out * w_out, hidden_dims[-1] * h_out * w_out),
                self.activation
            )
        )
        self.decoder_input = nn.Sequential(*modules)

        # Build Decoder
        modules = []
        hidden_dims.reverse()
        ind = 0
        h_in, w_in = h_out, w_out
        for index, i in enumerate(hidden_dims):
            try:
                out = hidden_dims[index + 1]
            except:
                out = 1
            h_out, w_out = contrans2d_layers(h_in, w_in, (0, 0), (2, 2), (2, 1))
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(i,
                                       out,
                                       kernel_size=(2, 2),
                                       stride=(2, 1),
                                       padding=(0, 0),
                                       dilation=(1, 1)
                                       ),
                    self.activation)
            )
            ind += 1
            h_in, w_in = h_out, w_out
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        mu1 = self.fc_mu1(mu)
        log_var = self.fc_var(mu)
        return [mu1, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 8, 25, 1)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input_: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input_)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input_, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class VAE2(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 6,
                 hidden_dims: List = None,
                 beta: int = 0.01,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type: str = 'H',
                 loss_weight: float = 3.,
                 l1: int = None,
                 l2: int = None,
                 activation=nn.Tanh(),
                 activation_latent=nn.Identity(),
                 **kwargs) -> None:
        super(VAE2, self).__init__()

        self.loss_weight = loss_weight
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter
        self.activation_latent = activation_latent  # nn.LeakyReLU()
        self.activation = activation
        modules = []
        if hidden_dims is None:
            hidden_dims = [16]  # , 32, 128, 256, 512
        if l1 is None or l2 is None:
            self.l1 = hidden_dims[-1] * 3 * 10
            self.l2 = hidden_dims[-1] * 3 * 10
        else:
            self.l1 = l1
            self.l2 = l2

        # Build Encoder
        ind = 0
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=(10-9*ind, 1+2*ind),
                              stride=(10-9*ind, 1),
                              padding=(0, 1*ind)
                              ),
                    # nn.BatchNorm2d(h_dim),
                    # nn.LeakyReLU())
                    self.activation)
            )
            in_channels = h_dim
            ind += 1
        self.encoder = nn.Sequential(*modules)

        # Build the middle linear layers:
        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1] * 3 * 10, hidden_dims[-1] * 3 * 10),
                # nn.BatchNorm1d(latent_dim),
                self.activation,
                nn.Linear(hidden_dims[-1] * 3 * 10, self.l1),
                # nn.BatchNorm1d(latent_dim),
                self.activation,
                nn.Linear(self.l1, latent_dim),
                self.activation_latent,
                nn.Linear(latent_dim, latent_dim),
            )
        )
        self.fc_mu = nn.Sequential(*modules)
        self.fc_mu1 = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        # Build Linear Layers Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                # nn.BatchNorm1d(latent_dim),
                self.activation_latent,
                nn.Linear(latent_dim, self.l2),
                # self.activation,
                # nn.Linear(hidden_dims[-1] * 3 * 10, self.l2),
                # nn.BatchNorm1d(hidden_dims[-1] * 3 * 10),
                self.activation,
                nn.Linear(self.l2, hidden_dims[-1] * 3 * 10),
            )
        )
        self.decoder_input = nn.Sequential(*modules)
        # Build Decoder
        modules = []
        hidden_dims.reverse()
        ind = 0
        for i in hidden_dims:  # range(len(hidden_dims) - 1)
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(i,    # hidden_dims[i]
                                       1,
                                       kernel_size=(10, 1),
                                       stride=(10, 1),
                                       # padding=(0, 1-ind),
                                       # output_padding=0
                                       ),
                    # nn.BatchNorm2d(hidden_dims[i + 1]),
                    self.activation)
                    # nn.LeakyReLU())
            )
            ind += 1
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        mu1 = self.fc_mu1(mu)
        log_var = self.fc_var(mu)

        return [mu1, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 16, 10, 3)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

