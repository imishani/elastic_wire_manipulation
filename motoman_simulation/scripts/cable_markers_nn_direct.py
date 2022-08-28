#!/usr/bin/env python

"""
Author: Itamar Mishani
Mail: imishani@gmail.com
"""

import rospy
import tf
import numpy as np
import pandas as pd
import rodeval
from std_msgs.msg import Float64MultiArray, Header
from visualization_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, PoseStamped, WrenchStamped, Wrench, Vector3
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
import torch, torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from scipy.spatial.transform import Rotation as R
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../learn_rod_path/itamar/train"))
from sklearn.preprocessing import StandardScaler


class cable_location_nn(object):
    def __init__(self, Q, ref_frame="LeftHand"): #left_cable_contact_pos
        super(cable_location_nn, self).__init__()
        self.ref_frame = ref_frame
        self.pub_line_min_dist = rospy.Publisher('~cable_line', Marker, queue_size=1)
        self.marker = Marker()
        self.marker.header.frame_id = ref_frame
        self.marker.type = self.marker.LINE_STRIP   # LINE_STRIP, SPHERE_LIST
        self.marker.action = self.marker.ADD

        # self.marker scale
        self.marker.scale.x = 0.02
        self.marker.scale.y = 0.02
        self.marker.scale.z = 0.02

        # self.marker color

        self.marker.scale.x = 0.02
        self.marker.color.r = 0.5
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.
        # self.marker orientaiton
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0

        # self.marker position
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0

        self.tl = tf.TransformListener()
        self.df = pd.DataFrame({'x': Q[:, 0], 'y': Q[:, 1], 'z': Q[:, 2]})
        rospy.sleep(1.2)
        self.A = self.tf_trans('base_link', 'left_cable_contact_pos') #arm_left_end_effector_link
        self.A_mocap = self.tf_trans('LeftHand', 'mocap')
        self.A_relative = self.tf_trans('left_cable_contact_pos', 'right_cable_contact_pos')

        self.df['poses'] = self.df.apply(lambda row: self.position(row), axis=1)

    def add_cable_to_rviz(self): # , dele=False
        self.marker.points = []
        for index, row in self.df[::-1].iterrows():
            pos = Point()
            pos.x, pos.y, pos.z = row.x, row.y, row.z
            self.marker.points.append(pos)
        self.pub_line_min_dist.publish(self.marker)

    def update_cable_pos(self, positions):
        self.df['x'] = positions[:, 0]
        self.df['y'] = positions[:, 1]
        self.df['z'] = positions[:, 2]
        self.df['poses'] = self.df.apply(lambda row: self.position(row), axis=1)

        self.add_cable_to_rviz() # True
        # rospy.loginfo('change approved!')

    def processFeedback(self, feedback):
        p = feedback.pose.position
        print(feedback.marker_name + " is now at " + str(p.x) + ", " + str(p.y) + ", " + str(p.z))

    def to_base_link(self, row):
        v1 = np.array([[row.x], [row.y], [row.z], [1]])
        v0 = np.dot(self.A[-1], v1)
        return Pose(Point(x=v0[0], y=v0[1], z=v0[2]),
                    Quaternion(x=0, y=0,
                               z=0, w=0))

    def position(self, row):
        transf = self.tf_trans('left_cable_contact_pos', 'right_cable_contact_pos')
        return Pose(Point(x=row.x, y=row.y, z=row.z),
                    Quaternion(x=transf[1][0], y=transf[1][1],
                               z=transf[1][2], w=transf[1][3]))

    def tf_trans(self, target_frame, source_frame):
        try:
            # listen to transform, from source to target. if source is 0 and tagret is 1 than A_0^1
            (trans, rot) = self.tl.lookupTransform(target_frame, source_frame, rospy.Time(0))
            mat = self.tl.fromTranslationRotation(trans, rot)
            return trans, rot, mat
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 'couldnt find mat', None, None

first_time = True
forces_register = None


def callback(data, model):

    forces = np.array([data.wrench.torque.z,
                       data.wrench.torque.x - 0.11*data.wrench.force.y,
                       data.wrench.torque.y - 0.11*data.wrench.force.x,
                       data.wrench.force.z,
                       data.wrench.force.x,
                       data.wrench.force.y])
    # if 2 mm:
    mu = np.array([-0.06990777, -0.21053933, -0.12817508, -3.40543442, -0.07793685, 2.78281192])
    std = np.array([0.02894388, 0.40500549, 0.32527397, 0.64293891, 1.02876501, 1.15996231])

    forces -= mu
    forces /= std

    global first_time, forces_register
    if first_time:
        forces_register = forces
        with torch.no_grad():
            if avishai:
                p_predict = model(torch.tensor(forces).float().view(1, 6))
            else:
                p_predict = model.decode(torch.tensor(forces).float().view(1, 6))
            p_predict = p_predict.view(-1, 3)
            p_predict = p_predict.numpy()
        Q = p_predict
        global cable_nn
        cable_nn = cable_location_nn(Q=Q)
        cable_nn.add_cable_to_rviz()
        first_time = False
    else:
        # Exponential filter:
        forces = 0.7*forces + 0.3*forces_register
        with torch.no_grad():
            if avishai:
                p_predict = model(torch.tensor(forces).float().view(1, 6))
            else:
                p_predict = model.decode(torch.tensor(forces).float().view(1, 6))
            p_predict = p_predict.view(-1, 3)
            p_predict = p_predict.numpy()
            # p_predict = p_predict[:int(100 * (65. / 82)), :]
        forces_register = forces
        Q2 = p_predict
        cable_nn.update_cable_pos(Q2)


class MLP(nn.Module):
    def __init__(self, input_dim=6, output_dim=100 * 3, n_layers=3, size=500,
                 device='cpu', dropout_p=0.2, activation=nn.LeakyReLU()):  # nn.LeakyReLU()
        super(MLP, self).__init__()
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_dim, size))
        self.mlp.append(nn.Tanh())
        self.mlp.append(nn.BatchNorm1d(size))
        for h in range(n_layers - 1):  # additional hidden layers
            self.mlp.append(nn.Linear(size, size))
            self.mlp.append(activation)
            self.mlp.append(nn.BatchNorm1d(size))
            self.mlp.append((nn.Dropout(p=dropout_p)))
        self.mlp.append(nn.Linear(size, output_dim))

        self.layers = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.layers(x)


class BaseVAE(nn.Module):

    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def sample(self, batch_size, current_device, **kwargs):
        raise NotImplementedError

    def generate(self, x, **kwargs):
        raise NotImplementedError


class SAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels=1,
                 latent_dim=6,
                 hidden_dims=None,
                 beta=0.1,
                 gamma=1000.,
                 max_capacity=25,
                 capacity_max_iter=1e5,
                 loss_type='H',
                 loss_weight=3.,
                 l1=None,
                 l2=None,
                 activation=nn.Tanh(),
                 activation_latent=nn.Identity(),
                 **kwargs):
        super(SAE, self).__init__()

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
                nn.Linear(latent_dim, latent_dim)
            )
        )
        self.fc_mu = nn.Sequential(*modules)
        self.fc_var = nn.Linear(hidden_dims[-1]*3*10, latent_dim)

        # Build Linear Layers Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                # nn.BatchNorm1d(latent_dim),
                self.activation_latent,
                nn.Linear(latent_dim, latent_dim*10),
                self.activation,
                nn.Linear(latent_dim*10, self.l2),
                self.activation,
                nn.Linear(self.l2, self.l2),
                # nn.BatchNorm1d(hidden_dims[-1] * 3 * 10),
                self.activation,
                nn.Linear(self.l2, hidden_dims[-1] * 3 * 10),
                self.activation
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

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # print('after encoder: ' + str(result.shape))
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        # print(f'z shape: {z.shape}')
        result = self.decoder_input(z)
        # print(f'result 1 {result.shape}')
        # Architecture 1:
        # result = result.view(-1, 32, 10, 3)
        # Architecture 2:
        result = result.view(-1, 16, 10, 3)
        # print(f'result 2 {result.shape}')
        result = self.decoder(result)
        # print(f'result 3 {result.shape}')
        # result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = mu    # self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def distance(self, p_pred, p):
        p_pred, p = p_pred.reshape(p_pred.shape[1], p_pred.shape[2]), p.reshape(p.shape[1], p.shape[2])
        d = torch.tensor([torch.sqrt(torch.sum((p[i, :] - p_pred[i, :])**2)) for i in range(p_pred.shape[0])])
        return torch.mean(d)

    def loss_function(self,
                      *args,
                      **kwargs):
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

    def loss_latent_constraints(self,
                                *args,
                                **kwargs):
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        force = kwargs['a']
        recons_loss = torch.sqrt(F.mse_loss(recons, input))

        # recons_loss = []
        # for i in range(recons.shape[0]):
        #     recons_loss.append(self.distance(recons[i, :, :], input[i, :, :]))
        # recons_loss = torch.mean(torch.tensor(recons_loss))

        ### Classic MSE Loss force and torques:
        a_loss_torque = torch.sqrt(F.mse_loss(mu[:, :3], force[:, :3]))  # , reduction="sum"
        a_loss_force = torch.sqrt(F.mse_loss(mu[:, 3:], force[:, 3:]))  # , reduction="sum"
        # loss = recons_loss + self.loss_weight * a_loss_torque + self.loss_weight * 1.2 * a_loss_force

        ### MSE and STD Loss:
        # lost = []
        # for i in range(force.shape[0]):
        #     lost.append(
        #         np.linalg.norm(force[i, :].to('cpu').detach().numpy() - mu[i, :].to('cpu').detach().numpy()))
        # rmse = np.sqrt(((force.to('cpu').detach().numpy() - mu.to('cpu').detach().numpy()) ** 2).mean(axis=0))
        # std = np.sqrt(((force.to('cpu').detach().numpy() - mu.to('cpu').detach().numpy()) ** 2).std(axis=0))
        # loss = recons_loss + self.loss_weight * np.mean(rmse) + self.loss_weight * np.mean(std)

        ### Using decoder to create same scale of loss:
        from_latent = self.decode(force)
        latent_loss = torch.sqrt(F.mse_loss(from_latent, input))
        # latent_loss = []
        # for i in range(from_latent.shape[0]):
        #     latent_loss.append(self.distance(from_latent[i, :, :], input[i, :, :]))
        # latent_loss = torch.mean(torch.tensor(latent_loss))
        loss = recons_loss + self.loss_weight*latent_loss + self.loss_weight * a_loss_torque + \
               self.loss_weight * a_loss_force
        # if self.num_iter % 1000 == 1:
        #     print(mu[0, :], force[0, :])
        #     print(f'Latent loss: {latent_loss}')
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'a torque': a_loss_torque, 'a force': a_loss_force,
                'latent loss': latent_loss}

    def sample(self,
               num_samples,
               current_device, **kwargs):
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

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


if __name__ == "__main__":
    rospy.init_node("nn_direct_marker")
    avishai = False
    elec = False
    path_file = 'check_point_2mm_10_7.pt'
    model = SAE()
    model.load_state_dict(torch.load(path_file))
    model.to('cpu')
    model.eval()
    a0 = rospy.Subscriber('/ft_compensated', WrenchStamped, callback, (model), queue_size=1)
    # r.sleep()
    rospy.spin()