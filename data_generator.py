import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import time
import random
import glob
import parse
import math
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from numpy.polynomial import Polynomial 
#from pathos.pools import ProcessPool
from scipy import linalg, interpolate
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from PIL import Image as im
from sklearn import gaussian_process as gp
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import h5py

# (only for stochastic)
#from stable_baselines3 import PPO
#from stable_baselines3.common.env_util import make_vec_env

from utils import redirect_log_file, timing
#from PDP.JinEnv import StochasticPendulumEnv
import oc_model as ocm
import utils
from utils import timing
from PDP import PDP, JinEnv
from config import Config

def fit_params(system_name, u, t, state_dict):
    config = Config()
    config.set_(None, system_name)
    u_model = ocm.get_u(config)
    if not state_dict is None:
        u_model.load_state_dict(state_dict)
    max_iter = int(5e3)
    min_loss = 1e-3
    lr = 0.01
    u = torch.tensor(u).float().squeeze()  #shape (n_sensor, u_dim)
    t = torch.tensor(t).float() 
    optimizer =  optim.Adam(u_model.parameters(), lr=lr)
    
    best_loss = 1e10
    
    no_update_iter, no_update_tole = 0, 100

    for i in range(max_iter):
        optimizer.zero_grad()
        u_pred = u_model(t)
        loss = F.mse_loss(u, u_pred)
        loss.backward()
        optimizer.step()
        no_update_iter += 1
        #if i % 100 == 0:
        #    print("epoch {}, loss {}".format(i, loss))
        if loss < min_loss:
            #print("epoch {}, quit with loss {}".format(i, loss))
            break
        elif loss < best_loss:
            best_loss = loss
            no_update_iter = 0
            #print("epoch {}, update loss {}".format(i, loss))
        elif no_update_iter >= no_update_tole:
            #print("epoch {}, quit with loss {}".format(i, loss))
            break
    params = torch.cat([p.data.detach().flatten() for p in u_model.parameters()])
    return params.numpy(), u_model.state_dict()




  
class OC_Data_Generator(object):

    def __init__(self, T, name, horizon=100):
        self.env, self.s0, self.loss_coef, self.sT, self.dyn_args = JinEnv.get_env(name)
        self.sT = np.array(self.sT)
        self.solver = PDP.OCSys()
        

        self.T = T
        self.horizon = horizon # the granularity of OC solver discretization
        self.name = name
    
    def gen_reg_grid(self, n_sample, s_offset, s_range, mode='t'):
        t_grid = np.linspace(0, self.T, num=self.horizon+1)[:,None] #shape=(self.horizon+1, t_dim)

        s_grid_list = list()

        if 'd' in mode:
            dyn_args = self.dyn_args + self.dyn_args*(np.random.rand(n_sample, len(self.dyn_args))*s_range + s_offset)*0.1  
            s_grid_list.append(dyn_args)

        if 't' in mode and 'i' in mode:
            ini_state = self.s0 + (np.random.rand(n_sample, len(self.s0))*s_range + s_offset) * 0.1
            s_grid_list.append(ini_state)
            diff = self.s0 - ini_state 
            if self.name in ["Quadrotor", "Rocket"]:
                mask = np.array([True]*len(self.s0))
                mask[6:10] = False  #remove the attitude dims
                diff = diff[:, mask]
            s_target = self.sT - diff  #the distance between ini_state and s_target is fixed for stablility
            s_grid_list.append(s_target)
        else:
            if 'i' in mode:
                ini_state = self.s0 + (np.random.rand(n_sample, len(self.s0))*s_range + s_offset)*0.1
                s_grid_list.append(ini_state)
            
            if 't' in mode:
                s_target = self.sT + np.random.rand(n_sample, len(self.sT))*s_range + s_offset  
                s_grid_list.append(s_target)

        s_grid = np.concatenate(s_grid_list, axis=1)  #shape=(n_sample, s_dim)
        return s_grid,t_grid
        

    @timing
    def gen_data(self, n_sample, s_offset, s_range, mode='t', n_sample_per_trial = 10):
        # num: number of 't'
        print("Generating data...", flush=True)

        samples = list()  #for each solution trajectory, sample 10 time indices
        while len(samples) <  n_sample:
            ocp_inst = list()
            if 'd' in mode:
                dyn_args = self.dyn_args + self.dyn_args*(np.random.rand(len(self.dyn_args))*s_range + s_offset)*0.1  
                ocp_inst.append(dyn_args)
            else:
                dyn_args = self.dyn_args
            
            if 't' in mode and 'i' in mode:
                ini_state = self.s0 + (np.random.rand(len(self.s0))*s_range + s_offset) * 0.1
                ocp_inst.append(ini_state)
                diff = self.s0 - ini_state 
                if self.name in ["Quadrotor", "Rocket"]:
                    mask = np.array([True]*len(self.s0))
                    mask[6:10] = False  #remove the attitude dims
                    diff = diff[mask]
                s_target = self.sT - diff  #the distance between ini_state and s_target is fixed for stablility
                ocp_inst.append(s_target)
            else:
                if 'i' in mode:
                    ini_state = self.s0 + (np.random.rand(len(self.s0))*s_range + s_offset)*0.1
                    ocp_inst.append(ini_state)
                else:
                    ini_state = self.s0
                
                if 't' in mode:
                    s_target = self.sT + np.random.rand(len(self.sT))*s_range + s_offset  
                    ocp_inst.append(s_target)
                else:
                    s_target = self.sT
            
            

            
            
            ocp_inst = np.concatenate(ocp_inst)
            
            self.env.initDyn(*dyn_args)
            self.env.setTargetState(s_target)
            self.solver.setStateVariable(self.env.X)
            self.solver.setControlVariable(self.env.U)
            self.solver.setPathCost(self.env.path_cost)
            
            T = self.T + np.random.rand(1) * self.T / self.horizon  #random T for flexible time.
            dt = T / self.horizon
            self.solver.setDyn(self.env.X + dt * self.env.f)
            sol = self.solver.ocSolver(ini_state=ini_state, horizon=self.horizon)

            t = np.arange(self.horizon) * dt
            u = sol['control_traj_opt']

            
            if n_sample_per_trial < self.horizon:
                idx = np.random.choice(self.horizon, n_sample_per_trial)
            else:
                idx = np.arange(self.horizon)
            for i in idx:
                samples.append((ocp_inst, t[i], u[i]))
        ocp_inst = np.array([s[0] for s in samples])
        t = np.array([s[1] for s in samples])[:,None]
        u = np.array([s[2] for s in samples])
        
        X = [ocp_inst, t]
        y = u
        return X, y
    
    



class DAE_Data_Generator(object):

    def __init__(self, T, name):

        self.env, self.s0, _, _ = JinEnv.get_env(name)
        self.solver = PDP.DAESys()
        self.solver.setStateVariable(self.env.X)
        self.solver.setControlVariable(self.env.U)

        self.T = T
        self.horizon = 100
        self.name = name

    @timing
    def gen_data(self, n_sample):
        # num: number of 't'
        print("Generating data...", flush=True)
        u_dim = self.solver.n_control

        samples = list()
        n_sample_per_trial = 10  #for each solution trajectory, sample 10 time indexes
        while len(samples) <  n_sample:
            T = self.T + np.random.rand(1) * self.T / self.horizon  #random T for flexible time.
            dt = T / self.horizon
            t = np.arange(self.horizon) * dt
            self.solver.setDyn(self.env.X + dt * self.env.f)
            u = self.get_grf_u(T, t, u_dim)
            sol = self.solver.solver(ini_state=self.s0, horizon=self.horizon, u=u)

            s = sol['state_traj_opt']

            idx = np.random.choice(self.horizon, n_sample_per_trial)
            for i in idx:
                samples.append((u, t[i], s[i]))
        u = np.array([i[0].flatten() for i in samples]) #(n_sample,  horizon*u_dim)
        
        t = np.array([i[1] for i in samples])[:,None]
        s = np.array([i[2] for i in samples])

        if self.name in ["Quadrotor", "Rocket"]:
            mask = np.array([True]*s.shape[1])
            mask[6:10] = False  #remove the attitude dims
            s = s[:,mask]
        
        X = [u, t]
        y = s
        return X, y
    
    @staticmethod
    def get_grf_u(T, t, u_dim):
        N = 1000
        x = np.linspace(0, T, num=N)
        #print(x.shape)
        K = gp.kernels.RBF(length_scale=1)(x)
        L = np.linalg.cholesky(K + 1e-13 * np.eye(N))
        # for multi-dim u?
        feature = np.dot(L, np.random.randn(N, u_dim)).T
        u = interpolate.interp1d(
                np.ravel(x), feature, kind="cubic", copy=False, assume_sorted=True, fill_value="extrapolate"
            )
        res = u(t).T            #shape (100, u_dim)
        return res


class Push_Data_Generator(object):
    def __init__(self, T, name):
        freq = 250
        self.T = T
        self.horizon = int(T * freq)
        #self.bias = 25  #discard first 25 samples
        self.name = name
    

    def preprocess_data(self, save=True):
        shape = self.name.split('_')[-1]
        h5_data_dir = "/home/fengmingquan/push_data/abs/{}".format(shape)
        file_path = "./data/DeepControl_Pushing_{}.npz".format(shape)
        if os.path.exists(file_path):
            data = np.load(file_path)
            s_target, u = data["s_target"], data["u"]

        else:
            print("Prepocessing data...", flush=True)
            h5_fname_pat = os.path.join(h5_data_dir, "*.h5")
            fname_list = glob.glob(h5_fname_pat)
            s_target_list = list()
            u_list = list()

            for fname in fname_list:
                # parse filename to get exp settings
                _, basename = os.path.split(fname)
                settings = parse.parse("motion_surface={surface}_shape={shape}_a={acc}_v={vel}_i={side}_s={pos}_t={angle}.h5", basename )
                #acc (acceleration in mm/s^2),
                #vel (velocity in mm/s),
                #side (side number, starting from 0),
                #pos (initial contact point position on that side, ranging from 0 to 1),
                #angle (initial contact angle in rad).
               
                #filter
                if float(settings['acc']) != 500 or abs(float(settings['angle'])) > 0.7:
                    continue
                if float(settings['pos']) <= 0.2 or float(settings['pos']) >= 0.8:
                    continue

                f1 = h5py.File(fname, 'r')
                
                # ---  construct u --- 
                

                #time_u = f1['ft_wrench'][self.bias: self.bias+self.horizon, 1:]  #u at time domain, shape=(horizon, u_dim) = (100,3)
                #time_u = self.normalize(time_u, axis=0)
                # encoder-1: FFT
                #freq_u = np.fft.rfft(time_u, axis=0) #transform u to freq domain
                #freq_u[int(len(freq_u) * 0.05):] = 0 #filter high-freq components
                #time_u = np.fft.irfft(freq_u, axis=0) #transform u back to time domain

                # encoder-2: polynomial
                #t = f1['ft_wrench'][self.bias: self.bias+self.horizon, 0] - f1['ft_wrench'][self.bias, 0]
                #for dim in range(3):
                #    time_u[:,dim] = Polynomial.fit(t, time_u[:,dim], deg=4)(t)

                # encoder-3: SG filter
                u_t = f1['ft_wrench'][:, 0]
                raw_u_x = f1['ft_wrench'][:, 1]
                u_x = savgol_filter(raw_u_x, window_length=61, polyorder=2, mode='nearest')
                raw_u_y = f1['ft_wrench'][:, 2]
                u_y = savgol_filter(raw_u_y, window_length=61, polyorder=2, mode='nearest')
                u_norm = np.sqrt(u_x**2 + u_y**2)

                # determin time window via displacement
                obj_pose_t = f1['object_pose'][:, 0]
                obj_pose_x = f1['object_pose'][:, 1] - f1['object_pose'][0, 1]
                obj_pose_y = f1['object_pose'][:, 2] - f1['object_pose'][0, 2]
                obj_pose_norm = np.sqrt(obj_pose_x**2 + obj_pose_y**2)
                ini_val, end_val = obj_pose_norm[0], obj_pose_norm[-1]
                ini_val_bar = ini_val + (end_val - ini_val)/100
                ini_idx = np.where(obj_pose_norm < ini_val_bar)[0][-1]
                ini_time = obj_pose_t[ini_idx]
                ini_time -= 0.05  #left shift 0.05s
                

                try:
                    ini_u_idx = np.where(u_t < ini_time)[0][-1]
                except IndexError:
                    print("Abnormal traj, skipped:",fname)
                    continue
                u_norm = u_norm[ini_u_idx : ini_u_idx+self.horizon]

                if u_norm.max() < 1:
                    print("Zero u, skipped:",fname)
                    continue

                u_list.append(u_norm) # shape=(horizon=110,)   u_dim=1(omitted)
                # ---  construct u finish --- 


                # --- construct s_target --- 
                s_target = []
                s_target.append(float(settings['pos']))
                s_target.append(float(settings['acc'])/1000)  #rescale unit of accleration and velocity to m/s^2 and m/s
                s_target.append(math.sin(float(settings['angle'])))
                s_target.append(math.cos(float(settings['angle'])))
                # mass
                # inertia
                # shape
                # friction

                #encode side
                side_encode = [0]*4
                side_idx = int(float(settings['pos']))
                side_encode[side_idx] = 1
                s_target.extend(side_encode)

                # encode object_pose
                # encoder-1: point-wise evaluation
                #n_sensor = 5
                #for time_idx in np.linspace(self.bias, self.bias+self.horizon, num=n_sensor+1)[1:]:  
                #    obj_pose = f1['object_pose'][int(time_idx), 1:] #the first col stores time, remove.
                #    tip_pose = f1['tip_pose'][int(time_idx), 1:]
                #    s_target.extend(list(obj_pose))
                #    s_target.extend(list(tip_pose))
                #s_target_list.append(s_target) #shape = (s_dim,) = (5 + n_sensor * 6,)

                #encoder-2: polynomial basis expansion
                #t = f1['object_pose'][self.bias: self.bias+self.horizon, 0] - f1['object_pose'][self.bias, 0] #shape=(horizon, )
                #obj_pose = f1['object_pose'][self.bias: self.bias+self.horizon, 1:] #the first col stores time, remove. shape=(horizon, 3)
                #obj_pose = self.normalize(obj_pose, axis=0)
                #for dim in range(3):
                #    coef = Polynomial.fit(t, obj_pose[:,dim], deg=4).coef#shape=(5,)
                #    s_target.extend(coef.tolist())

                                    
                #tip_pose = f1['tip_pose'][self.bias: self.bias+self.horizon, 1:]
                #tip_pose = self.normalize(tip_pose, axis=0)
                #for dim in range(3):
                #    coef = Polynomial.fit(t, tip_pose[:,dim], deg=4).coef  #shape=(5,)
                #    s_target.extend(coef.tolist())

                # encoder-3: point-wise eval
                #ini_idx = np.where(obj_pose_t < ini_time)[0][-1]
                #n_sample = 50
                #obj_pose_idx_downsp = np.linspace(0, self.horizon, n_sample, dtype=int, endpoint=False)
                #obj_pose_norm = obj_pose_norm[ini_idx + obj_pose_idx_downsp]
                #s_target.extend(obj_pose_norm.tolist())

                s_target_list.append(s_target)
                # --- construct s_target finishs --- 
                
                

            s_target = np.array(s_target_list) #shape=(len(fname_list), s_dim)
            u = np.array(u_list) #shape=(len(fname_list), horizon)   u_dim=1(omitted)

            if save:
                np.savez_compressed(file_path, s_target=s_target, u=u)

        return s_target, u



    def gen_data(self, n_sample, s_offset=0.1, s_range=0.9, n_sample_per_trial = None, space="Equi"):
        print("Generating data...", flush=True)
        s_target, u = self.preprocess_data()
        # only sample data whose 'pos' in (s_offset,  s_offset + s_range]
        mask = (s_target[:, 0] > s_offset) * (s_target[:, 0] <= s_offset + s_range)
        s_target, u = s_target[mask], u[mask]
        n_prob = len(s_target)
        
        if n_sample_per_trial is None:
            n_sample_per_trial = max(3, math.ceil(n_sample / n_prob))

        print("n_prob={}, n_sample={}, n_sample_per_trial={}".format(n_prob, n_sample ,n_sample_per_trial))

        random.seed(0)
        np.random.seed(1)
        prob_idx_list = list(range(n_prob))
        random.shuffle(prob_idx_list) 
        samples = list()
        t = np.linspace(0, self.T, self.horizon+1)[1:]
        

        for prob_i in prob_idx_list:
            s_target_i = s_target[prob_i]
            u_i = u[prob_i]

            
            
            if n_sample_per_trial < self.horizon:
                if space == "Cheb":
                    shifted_t = (t / self.T) * 2 - 1  #shift t from [0,T] to [-1,1]
                    prob =  (1 - t**2) ** -0.5
                    prob /= np.sum(prob)
                    time_idx_list = np.random.choice(self.horizon, n_sample_per_trial, replace=False, p = prob)
                else:
                    time_idx_list = np.random.choice(self.horizon, n_sample_per_trial, replace=False)
            else:
                time_idx_list = np.arange(self.horizon)

            for time_j in time_idx_list:
                samples.append((s_target_i,  t[time_j], u_i[time_j]))
            
            if len(samples)>= n_sample:
                break

        s_target = np.array([s[0] for s in samples])
        t = np.array([s[1] for s in samples])[:,None]
        u = np.array([s[2] for s in samples])

        X = [s_target, t]   # np array (n_sample, s_dim), (n_sample, t_dim)
        y = u  # np array (n_sample, u_dim)
        return X, y
        

class Push_All_Data_Generator(object):
    def __init__(self, T, name):
        freq = 250
        self.T = T
        self.horizon = int(T * freq)
        #self.bias = 25  #discard first 25 samples
        self.name = name
    

    def preprocess_data(self, save=True):
        h5_data_dir = "/home/fengmingquan/push_data/"
        file_path = "./data/DeepControl_Pushing_all.npz"
        shape_img_dir = "/home/fengmingquan/push_data/shape/"
        if os.path.exists(file_path):
            data = np.load(file_path)
            s_target, u = data["s_target"], data["u"]

        else:
            print("Prepocessing data...", flush=True)
            s_target_list = list()
            u_list = list()
            mass_and_inertia = {
                'rect1': [0.837, 1.13],
                'rect2': [1.045, 1.81],
                'rect3': [1.251, 2.74],
                'hex'  : [0.983, 1.50],
                'ellip1':[0.894, 1.23],
                'ellip2':[1.110, 1.95],
                'ellip3':[1.334, 2.97],
                'butter':[1.197, 2.95],
                'tri1' : [0.803, 1.41],
                'tri2' : [0.983, 2.11],
                'tri3' : [1.133, 2.96],
            }
            friction_mean_std = {
                'abs': [0.15663390913318725, 0.016408766306536263],
                'pu' : [0.34908703456687235, 0.05224241252979857],
                'plywood':[0.25227581442518765, 0.016688902499577316],
                'delrin':[0.1519035754316195, 0.00993995664188148]
            }
            surfaces = ['abs','pu', 'plywood', 'delrin']
            shapes = ['butter', 'ellip1', 'ellip2', 'ellip3', 'hex', 'rect1', 'rect2','rect3','tri1', 'tri2', 'tri3']
            for surface_idx, surface in enumerate(surfaces):
                for shape_idx, shape in enumerate(shapes):
                    #shape .png files
                    img_fname = "{}{}.png".format(shape_img_dir, shape)
                    img = im.open(img_fname).convert("LA").resize((24, 32)) 
                    img_arr = np.asarray(img)
                    img_arr = img_arr[:,:,1] / 255
                    img_arr = img_arr.reshape((-1,)) #flatten to 1-d array


                    #trajectory .h5 files
                    h5_fname_pat = os.path.join(h5_data_dir, "{}/{}_h5/*.h5".format(surface, shape))
                    fname_list = glob.glob(h5_fname_pat)
                    

                    for fname in fname_list:
                        # parse filename to get exp settings
                        _, basename = os.path.split(fname)
                        settings = parse.parse("motion_surface={surface}_shape={shape}_a={acc}_v={vel}_i={side}_s={pos}_t={angle}.h5", basename )
                        #acc (acceleration in mm/s^2),
                        #vel (velocity in mm/s),
                        #side (side number, starting from 0),
                        #pos (initial contact point position on that side, ranging from 0 to 1),
                        #angle (initial contact angle in rad).
                    
                        #filter
                        if float(settings['acc']) != 500 or abs(float(settings['angle'])) > 0.7 or float(settings['side'])!=0:
                            continue
                        if float(settings['pos']) <= 0.2 or float(settings['pos']) >= 0.8:
                            continue

                        f1 = h5py.File(fname, 'r')
                        
                        # ---  construct u --- 

                        # encoder-3: SG filter
                        u_t = f1['ft_wrench'][:, 0]
                        raw_u_x = f1['ft_wrench'][:, 1]
                        u_x = savgol_filter(raw_u_x, window_length=61, polyorder=2, mode='nearest')
                        raw_u_y = f1['ft_wrench'][:, 2]
                        u_y = savgol_filter(raw_u_y, window_length=61, polyorder=2, mode='nearest')
                        u_norm = np.sqrt(u_x**2 + u_y**2)

                        # determin time window via displacement
                        obj_pose_t = f1['object_pose'][:, 0]
                        obj_pose_x = f1['object_pose'][:, 1] - f1['object_pose'][0, 1]
                        obj_pose_y = f1['object_pose'][:, 2] - f1['object_pose'][0, 2]
                        obj_pose_norm = np.sqrt(obj_pose_x**2 + obj_pose_y**2)
                        ini_val, end_val = obj_pose_norm[0], obj_pose_norm[-1]
                        ini_val_bar = ini_val + (end_val - ini_val)/100
                        ini_idx = np.where(obj_pose_norm < ini_val_bar)[0][-1]
                        ini_time = obj_pose_t[ini_idx]
                        ini_time -= 0.05  #left shift 0.05s
                        try:
                            ini_u_idx = np.where(u_t < ini_time)[0][-1]
                        except IndexError:
                            print("Abnormal traj, skipped:",fname)
                            continue
                        u_norm = u_norm[ini_u_idx : ini_u_idx+self.horizon]

                        if u_norm.max() < 1:
                            print("Zero u, skipped:",fname)
                            continue

                        u_list.append(u_norm) # shape=(horizon=110,)   u_dim=1(omitted)
                        # ---  construct u finish --- 


                        # --- construct s_target --- 
                        s_target = []
                        # +++ s_target part-1: exp settings +++ 
                        s_target.append(float(settings['pos']))
                        s_target.append(math.sin(float(settings['angle'])))
                        s_target.append(math.cos(float(settings['angle'])))
                        s_target.extend(mass_and_inertia[shape]) #mass and inertia, len=2

                        #shape_one_hot = [0] * len(shapes)
                        #shape_one_hot[shape_idx] = 1
                        #s_target.extend(shape_one_hot) # shape encode, len=11

                        surface_one_hot = [0] * len(surfaces)
                        surface_one_hot[surface_idx] = 1
                        s_target.extend(surface_one_hot) #surface encode, len=4
                        s_target.extend(friction_mean_std[surface]) # friction, len=2


                        # +++ s_target part-2: traject encoding +++ 
                        #encode obj_pose via sg-filter
                        #down sample obj_t and obj_pose
                        n_sample = 50
                        horizon_downsp = 22
                        obj_idx_downsp = np.linspace(0, len(obj_pose_t), n_sample, dtype=int, endpoint=False)
                        obj_t_downsp = obj_pose_t[obj_idx_downsp]
                        delta_t = (obj_t_downsp[-1]-obj_t_downsp[0])/(len(obj_t_downsp)-1)
                        obj_x_downsp = obj_pose_x[obj_idx_downsp]
                        obj_x_downsp_2 = -savgol_filter(obj_x_downsp, window_length=9, polyorder=5, deriv=2, delta=delta_t )
                        obj_y_downsp = obj_pose_y[obj_idx_downsp]
                        obj_y_downsp_2 = -savgol_filter(obj_y_downsp, window_length=9, polyorder=5, deriv=2, delta=delta_t )
                        obj_norm = np.sqrt(obj_x_downsp_2**2 + obj_y_downsp_2**2)
                        ini_obj_idx = np.where(obj_t_downsp < ini_time)[0][-1]
                        obj_norm = obj_norm[ini_obj_idx: ini_obj_idx+horizon_downsp]
                        s_target.extend(obj_norm.tolist())

                        # +++ s_target part-3: shape img +++
                        s_target.extend(img_arr.tolist())


                        s_target_list.append(s_target)
                        # --- construct s_target finishs --- 


            s_target = np.array(s_target_list) #shape=(len(fname_list), s_dim)
            u = np.array(u_list) #shape=(len(fname_list), horizon)   u_dim=1(omitted)

            if save:
                np.savez_compressed(file_path, s_target=s_target, u=u)

        return s_target, u



    def gen_data(self, n_sample, s_offset=0.1, s_range=0.9, n_sample_per_trial = None, sampling=True, space="Equi"):
        print("Generating data...", flush=True)
        s_target, u = self.preprocess_data()
        # only sample data whose 'pos' in (s_offset,  s_offset + s_range]
        mask = (s_target[:, 0] > s_offset) * (s_target[:, 0] <= s_offset + s_range)
        s_target, u = s_target[mask], u[mask]
        n_prob = len(s_target)
        
        if n_sample_per_trial is None:
            n_sample_per_trial = max(3, math.ceil(n_sample / n_prob))

        print("n_prob={}, n_sample={}, n_sample_per_trial={}, space={}.".format(n_prob, n_sample ,n_sample_per_trial, space))

        random.seed(n_sample)
        np.random.seed(n_sample_per_trial)
        prob_idx_list = list(range(n_prob))
        random.shuffle(prob_idx_list) 
        samples = list()
        t = np.linspace(0, self.T, self.horizon+1)[1:]
        

        for prob_i in prob_idx_list:
            s_target_i = s_target[prob_i]
            u_i = u[prob_i]

            
            
            if n_sample_per_trial < self.horizon and sampling:
                if space == "Cheb":
                    shifted_t = (t / self.T) * 2 - 1  #shift t from [0,T] to [-1,1]
                    prob =  (1 - shifted_t**2 + 1e-5) ** (-0.5)
                    prob /= np.sum(prob)
                    time_idx_list = np.random.choice(self.horizon, n_sample_per_trial, replace=False, p = prob)
                else:
                    time_idx_list = np.random.choice(len(u_i), n_sample_per_trial, replace=False)
            else:
                time_idx_list = np.arange(len(u_i))

            for time_j in time_idx_list:
                samples.append((s_target_i,  t[time_j], u_i[time_j]))
            
            if len(samples)>= n_sample:
                break

        s_target = np.array([s[0] for s in samples])
        t = np.array([s[1] for s in samples])[:,None]
        u = np.array([s[2] for s in samples])

        X = [s_target, t]   # np array (n_sample, s_dim), (n_sample, t_dim)
        y = u  # np array (n_sample, u_dim)
        return X, y
    

class Brachistochrone_Data_Generator(object):
    def __init__(self, T, name):
        self.T = T
        self.name = name
        self.horizon = 101
        self.dx = self.T/(self.horizon-1)
        self.sys = PDP.BrachistochroneSys()
    
    @timing
    def gen_data(self, n_sample, s_offset, s_range, n_sample_per_trial = 10, sampling=True):
        # num: number of 't'
        print("Generating data...", flush=True)

        samples = list()  #for each solution trajectory, sample 10 time indices
        while len(samples) <  n_sample:
            left_state, right_state = 2, 1
            left_state += np.random.rand(1)*s_range + s_offset  
            right_state += np.random.rand(1)*s_range + s_offset  
            ocp_inst = np.array([left_state, right_state]).squeeze()
            _, k, Theta = self.sys.get_instance(ini_state=ocp_inst, x_bound=self.T)
            sol = self.sys.solver(ini_state=ocp_inst, horizon=self.horizon-1, dx=self.dx)
            u = np.array(sol['control_traj_opt'])
            t = np.array(sol["x_idx"])
            
            
            if n_sample_per_trial < len(u) and sampling:
                idx = np.random.choice(len(u), n_sample_per_trial)
            else:
                idx = np.arange(len(u))
            for i in idx:
                samples.append((ocp_inst, t[i], u[i]))
        ocp_inst = np.array([s[0] for s in samples])
        t = np.array([s[1] for s in samples])[:,None]
        u = np.array([s[2] for s in samples])
        
        X = [ocp_inst, t]
        y = u
        return X, y


class Zermelo_Data_Generator(object):
    def __init__(self, T, name):
        self.T = T
        self.name = name
        self.horizon = 101
        self.dt = self.T/(self.horizon-1)
        self.sys = PDP.ZermeloSys()
    
    @timing
    def gen_data(self, n_sample, s_offset, s_range, n_sample_per_trial = 10, sampling=True):
        # num: number of 't'
        print("Generating data...", flush=True)

        samples = list()  #for each solution trajectory, sample 10 time indices
        n_trial = 0
        n_failure = 0
        while len(samples) <  n_sample:
            base_state = [1,1,3,0,0,0,0]
            ocp_inst = np.array(base_state) + np.random.rand(len(base_state))*s_range + s_offset  
            
            sol = self.sys.solver(ini_state=ocp_inst, horizon=self.horizon-1)
            u = np.array(sol['control_traj_opt'])
            t = np.arange(0, len(u)+1) * self.dt #Note: this is NOT the actually 'time' of u, but a dummy positional encoding.
            n_trial += 1

            # check correctness
            ana_u = self.sys.ana_solver(ini_state=ocp_inst, horizon=self.horizon-1)
            mape  = np.sum(abs(ana_u- u)) / np.sum(abs(ana_u))
            if mape > 0.05:
                n_failure +=1
                continue
            
            if n_sample_per_trial < len(u) and sampling:
                idx = np.random.choice(len(u), n_sample_per_trial)
            else:
                idx = np.arange(len(u))
            for i in idx:
                samples.append((ocp_inst, t[i], u[i]))
        ocp_inst = np.array([s[0] for s in samples])
        t = np.array([s[1] for s in samples])[:,None]
        u = np.array([s[2] for s in samples])
        
        X = [ocp_inst, t]
        y = u
        print("failure rate={}/{}={}".format(n_failure, n_trial, n_failure/n_trial))
        return X, y


def get_reg_grid(system_name, T, num_sample, s_offset=0.1, s_range=1, mode='t'):
    if system_name == "Pushing":
        generator = Push_All_Data_Generator(T, system_name)
        s_grid,t_grid = generator.gen_reg_grid(num_sample, s_offset, s_range)
    elif system_name.startswith("Brachistochrone"):
        generator = Brachistochrone_Data_Generator(T, system_name)
        s_grid,t_grid = generator.gen_reg_grid(num_sample, s_offset, s_range)
    elif system_name.startswith("Zermelo"):
        generator = Zermelo_Data_Generator(T, system_name)
        s_grid,t_grid = generator.gen_reg_grid(num_sample, s_offset, s_range)
    else: # Traditional OC systems [Pendulum,...]    
        generator = OC_Data_Generator(T, system_name)
        s_grid,t_grid = generator.gen_reg_grid(num_sample, s_offset, s_range, mode=mode)
    
    return torch.tensor(s_grid).float(), torch.tensor(t_grid).float()
    

def get_data(model_name, system_name, T, num_sample, s_offset=0.1, s_range=1, mode='t', space="equi" , save=True):
    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if "Control" in model_name:
        data_name = "DeepControl"  #DCN and MLPControl use same data
    elif model_name in ["DeepONet2", "MLPOp"]:  #DON2 and MLPOp use same data
        data_name = "DeepOp"
    else:
        data_name = model_name
    if mode == 't':
        name = "{}_{}_{}_{}_{}.npz".format(data_name, system_name, num_sample, s_offset, s_range)
    else:
        mode = ''.join(sorted(mode))
        name = "{}_{}_{}_{}_{}_{}.npz".format(data_name, system_name, mode, num_sample, s_offset, s_range)
    file_path = os.path.join(data_path, name)
    if os.path.exists(file_path):
        data = np.load(file_path)
        X, y = data["X"], data["y"]
    else:
        if system_name.startswith("Stochastic"):
            generator = StoOC_Data_Generator()
            X,y = generator.gen_train_data(num_sample, s_offset, s_range)
        elif system_name == "Heating":
            generator = PdeOC_Data_Generator(T, system_name)
            X,y = generator.gen_data(num_sample, s_offset, s_range)
        elif system_name == "Pushing":
            generator = Push_All_Data_Generator(T, system_name)
            X,y = generator.gen_data(num_sample, s_offset, s_range)
        elif system_name.startswith("Pushing_"):
            generator = Push_Data_Generator(T, system_name)
            X,y = generator.gen_data(num_sample, s_offset, s_range)
        elif system_name.startswith("Brachistochrone"):
            generator = Brachistochrone_Data_Generator(T, system_name)
            X,y = generator.gen_data(num_sample, s_offset, s_range)
        elif system_name.startswith("Zermelo"):
            generator = Zermelo_Data_Generator(T, system_name)
            X,y = generator.gen_data(num_sample, s_offset, s_range)
        else: # Traditional OC systems [Pendulum,...]
            if data_name == "DeepControl":
                generator = OC_Data_Generator(T, system_name)
                X, y = generator.gen_data(num_sample, s_offset, s_range, mode=mode)
 
        X = np.concatenate(X, axis=1)

        if save:
            np.savez_compressed(file_path, X=X, y=y)
    
    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float().squeeze())
    
    return dataset



