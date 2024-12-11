import warnings
warnings.filterwarnings("ignore")
import os
import sys
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np


from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv, GCNConv

from  config import Config

class LagPolyU(nn.Module):
    """  u(t)=LagrangePolynomial(t; pivots)"""
    def __init__(self, T, order, u_dim):
        super(LagPolyU, self).__init__()
        self.pivots = torch.linspace(0, T, steps=order)
        self.linear = nn.Linear(order, u_dim, bias=False)
        self.name = "LagPolyU" 
        self.order = order
    def forward(self, sensors):
        #sensors shape (n_sensor,)
        features = []
        for i in range(len(self.pivots)):
            bi = 1
            for j in range(len(self.pivots)):
                if j != i:
                    bi = bi * (sensors - self.pivots[j]) / (self.pivots[i] - self.pivots[j])
            features.append(bi)
        input_ = torch.vstack(features).T  #shape(n_sensor, order)
        output = self.linear(input_).squeeze() #output shape (n_sensor, u_dim)
        return output
        

class DirectU(nn.Module):
    """ directly output u(t)=params"""
    def __init__(self, n_sensor):
        super(DirectU, self).__init__()
        self.params = Parameter(torch.rand(n_sensor))
        self.name = "DirectU" 
        self.order = 0
    def forward(self, sensors):
        #input shape (n_sensor,)
        #output shape (n_sensor,)
        return self.params
    
class PolyU(nn.Module):
    """ Polynomial basis u(t)=[1, t, t^2, ...].dot(theta)"""
    def __init__(self, order):
        super(PolyU, self).__init__()
        self.order = order
        self.linear = nn.Linear(self.order, 1)
        self.name = "PolyU" 
    def forward(self, sensors):
        #sensors shape (n_sensor,)
        features = [sensors, ]
        for i in range(2, self.order+1):
            features.append(sensors ** i)
        input_ = torch.vstack(features).T  #shape(n_sensor, order)
        output = self.linear(input_).squeeze() #output shape (n_sensor,)
        return output

class FourierU(nn.Module):
    """ Fourier basis u(t)= [1+ sin(pi * t) + cos(pi * t) + sin(2pi * t) + ...].T beta """
    def __init__(self, order):
        super(FourierU, self).__init__()
        self.order = order
        self.linear = nn.Linear(self.order*2, 1)
        self.name = "FourierU" 
    def forward(self, sensors):
        #sensors shape (n_sensor,)
        features = []
        for i in range(1, self.order+1):
            features += [torch.sin(math.pi * i * sensors), torch.cos(math.pi * i * sensors) ]
        input_ = torch.vstack(features).T  #shape(n_sensor, order*2)
        output = self.linear(input_).squeeze() #output shape (n_sensor,)
        return output

class MLPU(nn.Module):
    """ MLP u(t) = MLP(t) """
    def __init__(self, order, u_dim):
        super(MLPU, self).__init__()
        self.order = order
        width = 4
        t_dim = 1
        width_list = [t_dim] + [width]*order + [u_dim]
        self.mlp = MLP(width_list)
        self.name = "MLPU" 
    def forward(self, sensors):
        #sensors shape (n_sensor,)
        input_ = torch.unsqueeze(torch.tensor(sensors), 1)  #shape(n_sensor, 1)
        output = self.mlp(input_).squeeze() #output shape (n_sensor, u_dim)
        return output

class MLP(nn.Module):
    '''MLP Backbone for all models'''
    def __init__(self, width_list, output_activated=False):
        super(MLP, self).__init__()
        layers = []
        self.width_list = width_list
        for i in range(len(width_list)-1):
            layers.append(nn.Linear(width_list[i], width_list[i+1]))
            layers.append(nn.ReLU())
        if not output_activated:
            layers.pop(-1) # last layer is not activated
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    '''CNN Backbone for Pushing_img dataset'''
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(144, 20)  
        self.fc2 = nn.Linear(20, 11)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)  #shape= (batch_size, 11)
        return x

    def encode(self, s, img_size):
        img_len = img_size[0] * img_size[1]
        exp_setting, img =  s[:, :s.shape[1]-img_len], s[:, -img_len:]
        img = img.reshape((-1, 1, img_size[0], img_size[1]))  #1-channel grayscale img 
        img_encode = self.forward(img)
        s_encode = torch.cat((exp_setting, img_encode), dim=1)
        return s_encode

class Chebshev():
    def __init__(self, order, T):
        self.order = order
        self.T = T
    
    def forward(self, t):
        ''' Chebshev series of the first kind.
            input shape (B, t_dim=1)
            output shape (B, self.order+1)
        ''' 
        # scale and shift input from [0,T] to [-1, 1]
        t = t * 2 / self.T - 1

        
        
        # forward (iterative implementation)
        #output_list = list()
        #for o in range(self.order+1):
        #    if o == 0:
        #        output = torch.ones_like(t)
        #    elif o == 1:
        #        output = t
        #    else:
        #        output = 2 * t * output_list[o-1] - output_list[o-2]  #recurrence definition
        #    output_list.append(output)

        # forward (direct implementation)
        output_list = [torch.cos(o * torch.acos(t)) for o in range(self.order+1)]

        return torch.cat(output_list, dim=1)
    
    def adaptive_forward(self, t, shift, scale):
        ''' Chebshev series of the first kind.
            input shape t, shift, scale: (B, t_dim=1),   
            output shape (B, self.order+1)
        ''' 
        # scale and shift input from [0,T] to [-1, 1]
        t = t * 2 / self.T - 1

        output_list = [torch.cos((o + F.sigmoid(scale[:,o:o+1]) - 0.5) * torch.acos(t) + math.pi/2 * (F.sigmoid(shift[:,o:o+1]) - 0.5)) for o in range(self.order+1)]

        return torch.cat(output_list, dim=1)

class Fourier():
    def __init__(self, order, T, n_repeat=1):
        self.order = order
        self.T = T
        self.n_repeat = n_repeat
    
    def forward(self, t):
        ''' Fourier basis f(t)= [1+ sin(pi * t) + cos(pi * t) + sin(2pi * t) + ...]
            input shape (B, t_dim=1)
            output shape (B, self.order*2 + 1)
        ''' 
        t = t / self.T  #map [0,T] to [0,1]


        output_list = list()
        for o in range(self.order+1):
            if o == 0:
                output_list.append(torch.ones_like(t))
            else:
                output_list.append(torch.sin(math.pi * o * t))
                output_list.append(torch.cos(math.pi * o * t))
            
        return torch.cat(output_list, dim=1)
    
    def adaptive_forward(self, t, shift, scale):
        ''' Fourier basis f(t)= [1+ sin(pi * t) + cos(pi * t) + sin(2pi * t) + ...],
            but t is shifted and scaled before input to f().

            input shape t: (B, t_dim=1);   shift, scale: (B, Np)
            output shape (B, Np)
                where Np = 2* self.order + 1
        ''' 
        t = t / self.T #map [0,T] to [0,1]

        freqency = torch.arange(1, self.order+1)

        base = torch.ones_like(t)  #shape (B,1)
        sin_scale = F.sigmoid(scale[:, :self.order]) - 0.5  # shape (B, order)
        sin_shift = F.sigmoid(shift[:, :self.order]) - 0.5
        sin  = torch.sin(math.pi * (freqency + sin_scale) * t + math.pi/2 * sin_shift)
        cos_scale = F.sigmoid(scale[:, self.order: 2* self.order]) - 0.5  # shape (B, order)
        cos_shift = F.sigmoid(shift[:, self.order: 2* self.order]) - 0.5
        cos  = torch.cos(math.pi * (freqency + cos_scale) * t + math.pi/2 * cos_shift)
        
        return torch.cat((base, sin, cos), dim=1)  #shape=(B, 2*order+1)

    def repeat_adaptive_forward(self, t, shift, scale):
        ''' Fourier basis f(t)= [1+ sin(pi * t) + cos(pi * t) + sin(2pi * t) + ...],
            but t is shifted and scaled before input to f().

            input shape t: (B, t_dim=1);   shift, scale: (B, Np)
            output shape (B, Np)
                where Np = self.repeat * 2* self.order + 1
        ''' 
        t = t / self.T #map [0,T] to [0,1]

        freqency = torch.arange(1, self.order+1).repeat(self.n_repeat)  #shape= (self.order * self.repeat)

        base = torch.ones_like(t)  #shape (B,1)
        sin_scale = F.sigmoid(scale[:, :len(freqency)]) - 0.5  # shape (B, order)
        sin_shift = F.sigmoid(shift[:, :len(freqency)]) - 0.5
        sin  = torch.sin(math.pi * (freqency + sin_scale) * t + math.pi/2 * sin_shift)
        cos_scale = F.sigmoid(scale[:, len(freqency): 2* len(freqency)]) - 0.5  # shape (B, order)
        cos_shift = F.sigmoid(shift[:, len(freqency): 2* len(freqency)]) - 0.5
        cos  = torch.cos(math.pi * (freqency + cos_scale) * t + math.pi/2 * cos_shift)
        
        return torch.cat((base, sin, cos), dim=1)  #shape=(B, repeat*2*order+1)


class DeepControlNet5(nn.Module):
    """ DeepControlNet5, multiple outputs, and bias is a func of t
    """
    def __init__(self, branch_width, trunk_width, u_dim, img_size):
        super(DeepControlNet5, self).__init__()
        # branch & trunk nets
        self.branch_net = MLP(branch_width)
        trunk_width[-1] = branch_width[-1] + 1  #the additional 1 is bias
        self.trunk_net_list = [MLP(trunk_width) for _ in range(u_dim)]
        self.module_list = nn.ModuleList([self.branch_net])
        self.module_list.extend(self.trunk_net_list)

        # cnn encoder (only used in Pushing dataset)
        self.img_size = img_size
        if img_size[0] > 0:
            self.cnn = CNN()
            self.module_list.append(self.cnn)

        self.s_dim = branch_width[0]
        self.t_dim = trunk_width[0]
        self.u_dim = u_dim
        self.name = "DeepControlNet5"

    def forward(self, X1, X2=None):
        if X2 is None: #concatenated input form 
            s, t = X1[:, :-self.t_dim], X1[:, -self.t_dim:]
        else:  #splited input form
            s, t = X1, X2
        
        if self.img_size[0] > 0: #encode img
            s = self.cnn.encode(s, self.img_size)
        
        branch = self.branch_net(s)
        trunks_list = [self.trunk_net_list[i](t)[:,None,:] for i in range(self.u_dim)] #shape: [(batch_size, 1, trunk_width[-1]),... ], len= u_dim
        trunks = torch.cat(trunks_list, dim=1)# shape = (batch_size, u_dim, trunk_width[-1])
        trunks, bias = trunks[:,:, :-1], trunks[:,:, -1]
        y = torch.einsum("bi,bni->bn", branch, trunks) #inner product.  shape: (batch_size, u_dim)
        y = y + bias
        y = y.squeeze() #shape: (batch_size, u_dim)

        return y
    
    def forward_oc_solver(self, s, t, reshape=True):
        if self.img_size[0] > 0: #encode img
            s = self.cnn.encode(s, self.img_size)
        branch = self.branch_net(s) #shape = (n_prob, branch_width[-1]) 
        trunks_list = [self.trunk_net_list[i](t)[:,None,:] for i in range(self.u_dim)] #shape: [(n_sensor, 1, trunk_width[-1]),... ], len= u_dim,
        trunks = torch.cat(trunks_list, dim=1)# shape = (n_sensor, u_dim, trunk_width[-1]),  with batch_size = n_sensor
        trunks, bias = trunks[:,:, :-1], trunks[:,:, -1]  
        y = torch.einsum("pi,sui->psu", branch, trunks) #shape = (n_prob, n_sensor, u_dim)
        y = y+ bias[None,:,:]  #bias shape is (n_sensor, u_dim), broadcast 
        if reshape:
            #y.shape=(n_prob*n_sensor, u_dim)
            y = y.reshape((-1, self.u_dim))
            y = y.squeeze() 
        else:
            #y.shape=(n_prob, n_sensor, u_dim)
            pass
            
            
        return y 



class NSMControl6(nn.Module):
    """ Neural Spectral Method, version 6. 
        Compared with version5 
            shared coef_mat network, for saving num params 
    """
    def __init__(self, branch_width, trunk_width, T, u_dim, img_size, basis_func_type = 'Fourier', use_post_net=False):
        super(NSMControl6, self).__init__()
        
        # basis funcs
        self.basis_func_type = basis_func_type
        if basis_func_type == 'Chebshev':
            self.basis_func = Chebshev(order=branch_width[-1]-1, T=T)
        elif basis_func_type == 'Fourier':
            if branch_width[-1] % 2 == 0:
                branch_width[-1] += 1
            self.basis_func = Fourier(order=branch_width[-1]//2, T=T)

        #dims
        self.Np = branch_width[-1]
        self.s_dim = branch_width[0]
        self.t_dim = trunk_width[0]
        self.u_dim = u_dim

        # branch and trunk net
        branch_width[0] = self.s_dim + self.t_dim
        branch_width[-1] = 3 * self.Np
        self.branch_net_list = [MLP(branch_width) for _ in range(u_dim)] 
        
        self.module_list = nn.ModuleList(self.branch_net_list )

        #post-net
        self.use_post_net = use_post_net
        if self.use_post_net:
            self.post_net_list = [MLP([self.Np, 2* self.Np, 2*self.Np, 1]) for _ in range(u_dim)] 
            self.module_list.extend(self.post_net_list)

        # cnn encoder (only used in Pushing dataset)
        self.img_size = img_size
        if img_size[0] > 0:
            self.cnn = CNN()
            self.module_list.append(self.cnn)

        self.name = "NSMControl6"
        

    def forward(self, X1, X2=None):
        if X2 is None: #concatenated input form 
            s, t = X1[:, :-self.t_dim], X1[:, -self.t_dim:]
        else:  #splited input form
            s, t = X1, X2  #shape =(B, s_dim), (B, t_dim)
        
        if self.img_size[0] > 0: #encode img
            s = self.cnn.encode(s, self.img_size)
        
        u_list = list()
        s_t = torch.cat((s,t), dim=1)
        for i in range(self.u_dim):
            coef_mat = self.branch_net_list[i](s_t)  #shape=(B, 3 * Np)
            coef, shift, scale = coef_mat[:, :self.Np], coef_mat[:, self.Np:2*self.Np], coef_mat[:, 2*self.Np:3*self.Np]
            basis_mat = self.basis_func.adaptive_forward(t, shift, scale) #shape=(B,Np)
            if self.use_post_net:
                u = coef * basis_mat #shape=(B,Np)
                u = self.post_net_list[i](u)  #shape=(B,Np) --> (B,1)
            else:
                u = torch.einsum("bi,bi->b",  coef, basis_mat) #shape= (B,Np),(B,Np) --> (B,) inner product.            
            u_list.append(u)
        u = torch.stack(u_list, dim=1) #shape: (B, u_dim)
        u = u.squeeze() 

        return u



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_activated=True):
        super(ConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.output_activated = output_activated
        self.k = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same',)
        self.w = nn.Conv1d(in_channels, out_channels, 1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        

    def forward(self, x,):
        '''
        input shape x=(B, in_channel, x_hidden_dim), 
        output shape x=(B, out_channel, x_hidden_dim)
        '''
        x1 = self.k(x)
        x2 = self.w(x) #(B, out_channel, x_hidden_dim)
        x = x1 + x2
        if self.output_activated:
            x = F.gelu(x)

        return x

class ConvNet(nn.Module):
    def __init__(self, width, input_dim, out_channels, out_dim, kernel_size):
        super(ConvNet, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(input_dim, width)
        self.conv0 = ConvLayer(1, width, kernel_size)
        self.conv1 = ConvLayer(width, width, kernel_size)
        self.conv2 = ConvLayer(width, width, kernel_size)
        self.conv3 = ConvLayer(width, out_channels, kernel_size)
        self.fc1 = nn.Linear(width, out_dim)

    def forward(self, x,):
        '''
        input shape x=(B, input_dim), 
        output shape x=(B, out_dim, width)
        '''
        x = self.fc0(x)  #(B, input_dim) --> (B, width)
        x = x.reshape(-1, 1, self.width) #--> (B, 1, width)
        x = self.conv0(x) #(B, 1, width) --> (B, width, width)
        x = self.conv1(x) #no shape change
        x = self.conv2(x) #no shape change
        x = self.conv3(x) # (B, width, width) --> (B, out_channels, width)
        x = self.fc1(x) #(B, out_channels, width) --> (B, out_channels, out_dim) 

        return x


class NSMControl9(nn.Module):
    """ Neural Spectral Method, version 9. 
        Compared with version6 
            branch net is implemented by CNN instead of MLP
            add a post-transform at the network output
    """
    def __init__(self, branch_width, trunk_width, T, u_dim, img_size, basis_func_type = 'Fourier'):
        super(NSMControl9, self).__init__()
        
        # basis funcs
        self.basis_func_type = basis_func_type
        if basis_func_type == 'Chebshev':
            self.basis_func = Chebshev(order=branch_width[-1]-1, T=T)
        elif basis_func_type == 'Fourier':
            if branch_width[-1] % 2 == 0:
                branch_width[-1] += 1
            self.basis_func = Fourier(order=branch_width[-1]//2, T=T)

        #dims
        self.Np = branch_width[-1]
        self.s_dim = branch_width[0]
        self.t_dim = trunk_width[0]
        self.u_dim = u_dim

        # branch and trunk net
        branch_width[0] = self.s_dim + self.t_dim
        self.branch_net_list = [ConvNet(width=branch_width[1], input_dim=branch_width[0], out_channels=3, out_dim=self.Np, kernel_size=5) for _ in range(u_dim)] 
        self.post_net_list = [MLP([self.Np, 2* self.Np, 2*self.Np, 1]) for _ in range(u_dim)] 
        self.module_list = nn.ModuleList(self.branch_net_list + self.post_net_list )

        # cnn encoder (only used in Pushing dataset)
        self.img_size = img_size
        if img_size[0] > 0:
            self.cnn = CNN()
            self.module_list.append(self.cnn)

        self.name = "NSMControl9"
        

    def forward(self, X1, X2=None):
        if X2 is None: #concatenated input form 
            s, t = X1[:, :-self.t_dim], X1[:, -self.t_dim:]
        else:  #splited input form
            s, t = X1, X2  #shape =(B, s_dim), (B, t_dim)
        
        if self.img_size[0] > 0: #encode img
            s = self.cnn.encode(s, self.img_size)
        
        u_list = list()
        s_t = torch.cat((s,t), dim=1)
        for i in range(self.u_dim):
            coef_mat = self.branch_net_list[i](s_t)  #shape=(B, 3, Np)
            coef, shift, scale = coef_mat[:,0, :], coef_mat[:,1, :], coef_mat[:,2, :]  #shape=(B, Np)
            basis_mat = self.basis_func.adaptive_forward(t, shift, scale) #shape=(B,Np)
            u = coef * basis_mat #shape=(B,Np)
            u = self.post_net_list[i](u)  #shape=(B,Np) --> (B,1)
            u_list.append(u)
        u = torch.stack(u_list, dim=1) #shape: (B, u_dim)
        u = u.squeeze() 

        return u


class MLPControl(nn.Module):
    """ MLP directly maps [oc params, t] to u(t). 
    """
    def __init__(self, width_list, t_dim, img_size):
        super(MLPControl, self).__init__()
        self.mlp = MLP(width_list)
        self.name = "MLPControl"
        self.t_dim = t_dim

        # cnn encoder (only used in Pushing dataset)
        self.img_size = img_size
        if img_size[0] > 0:
            self.cnn = CNN()

    def forward(self, X1, X2=None):
        if X2 is None: #concatenated input form 
            s, t = X1[:, :-self.t_dim], X1[:, -self.t_dim:]
        else:  #splited input form
            s, t = X1, X2
            
        if self.img_size[0] > 0: #encode img
            s = self.cnn.encode(s, self.img_size)

        input_ = torch.cat((s, t), dim=1)
        u = self.mlp(input_).squeeze()
        return u #shape: (batch_size, u_dim) 

class MLPOp(nn.Module):
    """ MLP directly maps [u, t] to s(t). 
    """
    def __init__(self, width_list):
        super(MLPOp, self).__init__()
        self.mlp = MLP(width_list)
        self.name = "MLPOp"

    def forward(self, X1, X2=None):
        if X2 is None: #concatenated input form [u, t]
            input_ = X1
        else:  #splited input form
            input_ = torch.cat((X1, X2), dim=1)

        u = self.mlp(input_).squeeze()
        return u #shape: (batch_size, s_dim) 



#------------- FNO baseline  -------------#
# adapted from https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, input_dim, u_dim, t_dim, img_size):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        Original Input/Output:
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)

        Optimal Control Input/Output:
        input: target state x_goal, time index t
        input shape: (batchsize, input_dim), (batchsize, t_dim)
        output: the optimal control at time index t: u(t)
        output shape: (batchsize, u_dim)
        """

        self.modes1 = modes
        self.width = width
        self.input_dim = input_dim
        self.u_dim = u_dim
        self.t_dim = t_dim


        # cnn encoder (only used in Pushing dataset)
        self.img_size = img_size
        if img_size[0] > 0:
            self.cnn = CNN()

        self.fc0 = nn.Linear(input_dim+t_dim, self.width)
        
        self.conv0 = SpectralConv1d(1, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        
        self.w0 = nn.Conv1d(1, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        
        self.fc1 = nn.Linear(self.width**2, self.width)
        self.fc2 = nn.Linear(self.width, self.u_dim) 

        self.name = "FNOControl"

    def forward(self, X1, X2=None):
        if X2 is None: #concatenated input form 
            s, t = X1[:, :-self.t_dim], X1[:, -self.t_dim:]
        else:  #splited input form
            s, t = X1, X2
            
        if self.img_size[0] > 0: #encode img
            s = self.cnn.encode(s, self.img_size)

        x = torch.cat((s, t), dim=1)  #(batchsize, input_dim + t_dim)
        x = self.fc0(x)  #(batchsize, input_dim + t_dim) --> (batchsize, width)
        
        # map x to suitable shape for FFT.
        x = torch.unsqueeze(x, dim=1)  #(batchsize, width) --> (batchsize, 1, width)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)  #  (batchsize, 1, width)  --> (batchsize, width, width)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)  # (batchsize, width, width)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x) # (batchsize, width, width)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2   # (batchsize, width, width)

        x = x.flatten(start_dim=1) # (batchsize, width, width) --> # (batchsize, width**2)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  #x=fc2(fc1(x)): (batchsize, width**2) --> (batchsize, self.u_dim)
        x = torch.squeeze(x) #(batchsize, self.u_dim)
        return x


#------------- FNO baseline  ends-------------#


#------------- GEN baseline      -------------#
#https://github.com/FerranAlet/graph_element_networks/blob/master/GEN.py
#https://github.com/FerranAlet/graph_element_networks/blob/master/gen_softnn.py

class GEN(nn.Module):
    def __init__(self, n_nodes, width, input_dim, u_dim, t_dim, T, img_size):
        super(GEN, self).__init__()
        self.n_nodes = n_nodes
        self.width = width # num_features of hidden layers of GNN & MLP
        self.input_dim = input_dim
        self.u_dim = u_dim
        self.t_dim = t_dim # t_dim = position_dim
        self.T = T # horizon
        self.node_pos, self.edge_index = self.create_grid(n_nodes, t_dim, T)
        encoder_width_list = [input_dim, width, width, width]
        decoder_width_list = [width + t_dim, width, width, u_dim]
        self.encoder = MLP(encoder_width_list)
        self.decoder = MLP(decoder_width_list)
        self.msg_steps = int(math.sqrt(n_nodes))
        self.conv = GCNConv(width + t_dim, width) 
        self.layer_norm = nn.modules.normalization.LayerNorm(width)

        # cnn encoder (only used in Pushing dataset)
        self.img_size = img_size
        if img_size[0] > 0:
            self.cnn = CNN()

        self.name = "GENControl"

    def encode_and_msgpassing(self, s):        
        # Encode input
        if self.img_size[0] > 0: #encode img
            s = self.cnn.encode(s, self.img_size)
        encoded_s = self.encoder(s) #(batch_size, input_dim) --> (batch_size, width)

        # Create Batch to feed to GNN
        batch_size = encoded_s.shape[0]
        data_list = []
        for i in range(batch_size):
            # map encoded input to node features via softmax
            dist = (self.node_pos)**2 #shape=(n_nodes, t_dim)
            dist = torch.sqrt(dist.sum(dim=1, keepdim=True)) #(n_nodes, t_dim) --> (n_nodes, 1)
            weight = F.softmax(-dist, dim=0) #Softmax of negative distance 
            node_features = torch.mm(weight, encoded_s[i][None,:]) #(n_nodes, 1), (1, width)  -->  (n_nodes, width)
            data_list.append(Data(x=node_features, pos=self.node_pos, edge_index=self.edge_index))
        data = Batch.from_data_list(data_list)

        # Propagate GNN states with message passing
        for step in range(self.msg_steps):
            data.x = data.x + self.conv(torch.cat((data.pos, data.x), dim=-1), data.edge_index)
            data.x = self.layer_norm(data.x)
        
        #fetch the result
        batch_node_features = data.x.reshape((batch_size, self.n_nodes, self.width))
        batch_node_pos = data.pos.reshape((batch_size, self.n_nodes, self.t_dim))

        return batch_node_features, batch_node_pos

    def forward(self, X1, X2=None):
        if X2 is None: #concatenated input form 
            s, t = X1[:, :-self.t_dim], X1[:, -self.t_dim:]
        else:  #splited input form
            s, t = X1, X2

        # the shared part: encoder and msg passing
        batch_node_features, batch_node_pos = self.encode_and_msgpassing(s)
        
        # get hidden state via softmax aggragation of node features.
        dist = (batch_node_pos - t[:,None,:])**2 #shape(batch_size, n_nodes, t_dim)
        dist = torch.sqrt(dist.sum(dim=2, keepdim=True)) #(batch_size, n_nodes, t_dim) --> (batch_size, n_nodes, 1)
        weight = F.softmax(-dist, dim=1).transpose(1, 2) #Softmax of negative distance #(batch_size, n_nodes, 1)-->(batch_size, 1, n_nodes)
        hidden = torch.bmm(weight, batch_node_features).squeeze() #(batch_size, 1, n_nodes), (batch_size, n_nodes, width)  -->  (batch_size, width)
        hidden = torch.cat((hidden, t), dim=1)  #(batch_size,  width+t_dim)
        
        # Decode hidden states to final outputs
        y = self.decoder(hidden) #(batch_size, width+t_dim) --> (batch_size, u_dim)
        y = y.squeeze()
        return y
    
    def forward_oc_solver(self, X1, X2=None):
        # batched-query forward func, used for testing

        if X2 is None: #concatenated input form 
            s, t = X1[:, :-self.t_dim], X1[:, -self.t_dim:]
        else:  #splited input form
            s, t = X1, X2  
            # s.shape=(batch_size, input_dim), t.shape=(n_sensor, t_dim)

        # shared part: encoder and msg passing
        batch_node_features, batch_node_pos = self.encode_and_msgpassing(s)
        # batch_node_pos.shape = (batch_size, n_nodes, t_dim)

        # get hidden state via softmax aggragation of node features.
        dist = (batch_node_pos[:,:,None,:] - t)**2 #shape(batch_size, n_nodes, n_sensor, t_dim)
        dist = torch.sqrt(dist.sum(dim=3, keepdim=False)) #(batch_size, n_nodes, n_sensor, t_dim) --> (batch_size, n_nodes, n_sensor)
        weight = F.softmax(-dist, dim=1)
        weight = weight.transpose(1, 2) #(batch_size, n_nodes, n_sensor)-->(batch_size, n_sensor, n_nodes)
        hidden = torch.bmm(weight, batch_node_features).squeeze() #(batch_size, n_sensor, n_nodes), (batch_size, n_nodes, width)  -->  (batch_size, n_sensor, width)
        

        # Decode hidden states to final outputs
        batch_size = s.shape[0]
        t = torch.tile(t, (batch_size, 1, 1)) #(n_sensor, t_dim) --> (batch_size, n_sensor, t_dim)
        hidden = torch.cat((hidden, t), dim=2)  #(batch_size, n_sensor,  width+t_dim)
        hidden = hidden.reshape((-1, self.width+self.t_dim)) #(batch_size*n_sensor,  width+t_dim)
        y = self.decoder(hidden) #(batch_size*n_sensor, width+t_dim) --> (batch_size*n_sensor, u_dim)
        y = y.squeeze()
        return y

    
    def create_grid(self, n_nodes, t_dim, T):
        if t_dim == 1:
            node_pos = [i/(n_nodes-1) for i in range(n_nodes)]
            node_pos = torch.FloatTensor(node_pos).unsqueeze(-1)

            edge_index = []
            for i in range(1, n_nodes):
                edge_index.append(torch.LongTensor([i-1, i])) #forward direction
                edge_index.append(torch.LongTensor([i, i-1])) #backward direction
            edge_index = torch.stack(edge_index).T 

        elif t_dim == 2:
            side_len = int(math.sqrt(n_nodes))

            node_pos = []
            for i in range(side_len):
                for j in range(side_len):
                    node_pos.append(torch.FloatTensor([i/(side_len-1), j/(side_len-1)]))
            node_pos = torch.stack(node_pos)

            edge_index = []
            for i in range(1, side_len):
                for j in range(1, side_len):
                    edge_index.append(torch.LongTensor([side_len*i+j, side_len*(i-1)+j]))
                    edge_index.append(torch.LongTensor([side_len*(i-1)+j, side_len*i+j]))
                    edge_index.append(torch.LongTensor([side_len*i+(j-1), side_len*i+j]))
                    edge_index.append(torch.LongTensor([side_len*i+j, side_len*i+(j-1)]))
            edge_index = torch.stack(edge_index).T
        else:
            raise ValueError(t_dim)
        
        #Returns:
        #   node_pos: FloatTensor, shape=(n_nodes, t_dim)
        #   edge_index: LongTensor, shape=(2, n_edges)
        return node_pos, edge_index
#------------- GEN baseline  ends-------------#


def get_u(config):
    name = config.u_name
    if name == "DirectU":
        model = DirectU(config.n_sensor)
    elif name == "PolyU":
        model = PolyU(config.u_order)
    elif name == "FourierU":
        model = FourierU(config.u_order)
    elif name == "LagPolyU":
        model = LagPolyU(config.T, config.u_order, config.u_dim)
    elif name == "MLPU":
        model = MLPU(config.u_order, config.u_dim)
    return model

def get_model(config):
    name = config.model_name
    
    
    if name.startswith("DeepControlNet5"):
        model = DeepControlNet5(config.branch_width, config.trunk_width, config.u_dim, config.img_size)

    elif name.startswith("NSMControl"):
        if name == "NSMControl6":
            model = NSMControl6(config.branch_width, config.trunk_width, config.T, config.u_dim, config.img_size,  basis_func_type = 'Fourier') 
        elif name == "NSMControl6_1":
            model = NSMControl6(config.branch_width, config.trunk_width, config.T, config.u_dim, config.img_size,  basis_func_type = 'Chebshev')     
        elif name == "NSMControl6_2":
            model = NSMControl6(config.branch_width, config.trunk_width, config.T, config.u_dim, config.img_size,  basis_func_type = 'Fourier', use_post_net=True)     
        elif name == "NSMControl9":
            model = NSMControl9(config.branch_width, config.trunk_width, config.T, config.u_dim, config.img_size,  basis_func_type = 'Fourier')
        elif name == "NSMControl9_1":
            model = NSMControl9(config.branch_width, config.trunk_width, config.T, config.u_dim, config.img_size,  basis_func_type = 'Chebshev')     
    elif name.startswith("MLPControl"):
        model = MLPControl(config.mlp_width, config.t_dim, config.img_size)
    elif name == "FNOControl":
        # Original FNO has 287862 params
        # Reduce to 7802 params in OC
        
        modes = 8  #<= floor(width/2) + 1
        if config.system_name in ["Pendulum", "RobotArm", "CartPole", "Heating", "Pushing", "Brachistochrone", "Zermelo"]:
            width = 14
        elif config.system_name == "Pushing_rect1":
            modes = 6
            width = 10
        else:  #"Quadrotor", "Rocket", 
            width = 20
        #modes = 4
        #width = 7
        model = FNO1d(modes, width, config.input_dim, config.u_dim, config.t_dim, config.img_size)
    elif name == "GENControl":
        n_nodes = 9
        width = 20
        model = GEN(n_nodes, width, config.input_dim, config.u_dim, config.t_dim, config.T, config.img_size)
    else:
        raise NotImplementedError("The model named {} is not implemented".format(name))

    return model.to(config.device)


    test_nsm_9()