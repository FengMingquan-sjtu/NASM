import warnings
warnings.filterwarnings("ignore")
import os
import pickle
import math
import sys
import time
import argparse
import itertools
import contextlib
sys.path.append("./")

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR

from PDP import PDP, JinEnv
from config import Config
import data_generator as dg
import oc_model as ocm
import utils

def arr2ocp(arr, config):
    ret_dict = {"d":None, "i":None, 't':None}
    idx = 0

    for m in config.mode:
        if m == 't':
            dim = config.s_dim 
        elif m == 'i':
            dim = config.i_dim 
        elif m == 'd':
            dim = config.d_dim 
        val = arr[idx: idx+dim]
        ret_dict[m] = val 
        idx += dim
    
    assert idx == len(arr), "idx={} not equal to len(arr)={}".format(idx, len(arr))
    env, ini_state, loss_weight, x_goal, dyn_args = JinEnv.get_env(config.system_name)
    if not ret_dict["d"] is None:
        dyn_args = ret_dict["d"]
    if not ret_dict["i"] is None:
        ini_state = ret_dict["i"]
    if not ret_dict["t"] is None:
        x_goal = ret_dict['t']
    return x_goal, ini_state, dyn_args


def eval_heat_obj(u, system_name, s=None, goal=None):
    #----- create env ------------
    config = Config()
    config.set_(model_name=None, system_name=system_name)
    env, ini_state, loss_weight, _, _ = JinEnv.get_env(system_name)
    wu = loss_weight[-1] #weight of penalty
    dt = config.dt
    horizon = int(config.T / dt)

    # ----- solve  ------------
    solver = PDP.HeatDAESys()
    sol = solver.solver(ini_state=ini_state, horizon=horizon, dt=dt, x_goal=goal, alpha=wu, u=u)

    return sol['cost'], sol['control_traj_opt'], sol['state_traj_opt']

def eval_obj(u, system_name, s=None, goal=None, s0=None, dyn_args=None):
    
    if system_name == "Heating":
        return eval_heat_obj(u, system_name, s, goal)
    config = Config()
    env, s0_, loss_weight, x_goal, dyn_args_ = JinEnv.get_env(system_name)
    loss_coef = torch.tensor(loss_weight[:-1]).double()
    u_coef = loss_weight[-1]
    if system_name in ["Quadrotor", "Rocket"]:
        loss_coef = loss_coef.repeat_interleave(3)

    
    if s is None:  #solve DAE to get s
        if goal is None:
            goal = x_goal
        if dyn_args is None:
            dyn_args = dyn_args_
        if s0 is None:
            s0 = s0_
        env.initDyn(*dyn_args)
        solver = PDP.DAESys()
        solver.setStateVariable(env.X)
        solver.setControlVariable(env.U)
        solver.setDyn(env.X + config.dt * env.f)
        if torch.is_tensor(u):
            u = u.detach().numpy()#CasADI only accept np array of u
        sol = solver.solver(ini_state=s0, horizon=config.n_sensor, u=u)  
        s = sol['state_traj_opt'][:-1, :] #drop the last state for shape consistency
    
    
    if system_name in ["Quadrotor", "Rocket"]:
        mask = np.array([True]*s.shape[1])
        mask[6:10] = False  #remove the attitude dims
        s = s[:,mask]

    u, s, goal = torch.tensor(u), torch.tensor(s), torch.tensor(goal)
    obj = (s - goal).pow(2).sum(dim=0).dot(loss_coef)
    u_penalty =  u.pow(2).sum() * u_coef 
    obj += u_penalty

    
    return obj, sol['control_traj_opt'], sol['state_traj_opt'][:-1, :]


def direct_method(system_name, target_list, return_obj=True, mode='t', new_dt=None):
    #with contextlib.redirect_stdout(open(os.devnull, 'w')):
    #    with contextlib.redirect_stderr(open(os.devnull, 'w')):
    #----- create env ------------
    config = Config()
    config.set_(model_name=None, system_name=system_name, mode=mode)
    env, ini_state, loss_weight, _, _ = JinEnv.get_env(system_name)
    wu = loss_weight[-1] #weight of penalty
    dt = config.dt if new_dt is None else new_dt
    horizon = int(config.T / dt)

    #------ create solver --------
    if system_name == "Heating": #PDE-OC
        solver = PDP.HeatOCSys()
    elif system_name == "Brachistochrone":
        solver = PDP.BrachistochroneSys()
    elif system_name == "Zermelo":
        solver = PDP.ZermeloSys()
    
    else: #ODE-OC
        solver = PDP.OCSys()

    #------  solve ---------------
    obj_list = []
    state_list = []
    control_list = []
    start_time = time.time()
    
    for item in target_list:
        if system_name == "Heating": #PDE-OC
            s_target = item
            sol = solver.ocSolver(ini_state=ini_state, horizon=horizon, dt=dt, x_goal=s_target, alpha=wu)
            state_list.append(sol['state_traj_opt'])
            obj_list.append(sol['cost'])
            control_list.append(sol['control_traj_opt'])
        
        elif system_name == "Brachistochrone":
            solver.solver(ini_state=item, horizon=config.n_sensor, dx=config.dt)
        elif system_name == "Zermelo":
            solver.solver(ini_state=item, horizon=config.n_sensor)
        else: #ODE-OC
            s_target, ini_state, dyn_args = arr2ocp(item, config)
            env.initDyn(*dyn_args)
            solver.setStateVariable(env.X)
            solver.setControlVariable(env.U)
            solver.setDyn(env.X + dt * env.f)
            env.setTargetState(s_target)
            solver.setPathCost(env.path_cost)
            sol = solver.ocSolver(ini_state=ini_state, horizon=horizon)
            state_list.append(sol['state_traj_opt'][:-1, :])
            obj_list.append(sol['cost'])
            control_list.append(sol['control_traj_opt'])
    end_time = time.time()
    time_cost = end_time - start_time
    if return_obj:
        obj_arr = np.array(obj_list).squeeze()
        control_arr, state_arr = np.array(control_list), np.array(state_list)
        return obj_arr, control_arr, state_arr, time_cost
    else:
        return time_cost


def pdp(system_name, target_list, return_obj=True, mode='t'):
    #----- create env ------------
    config = Config()
    config.set_(None, system_name, mode)
    env, ini_state, loss_weight, _, _ = JinEnv.get_env(system_name)
    wu = loss_weight[-1] #weight of penalty
    dt = config.dt
    horizon = int(config.T / dt)


    #------ create solver --------
    pdp_solver = PDP.ControlPlanning()
    

    
    #------ set hyper params -----
    lr = 1e-3
    max_iter = 2500

    #------  solve ---------------
    obj_list = []
    state_list, control_list = [], []
    start_time = time.time()
    for item in target_list:
        s_target, ini_state, dyn_args = arr2ocp(item, config)
        env.initDyn(*dyn_args)
        env.setTargetState(s_target)
        pdp_solver.setStateVariable(env.X)
        pdp_solver.setControlVariable(env.U)
        pdp_solver.setDyn(env.X + dt * env.f)
        pdp_solver.setPathCost(env.path_cost)
        pdp_solver.init_step(horizon)
        initial_parameter = np.random.randn(pdp_solver.n_auxvar)
        current_parameter = initial_parameter
        for k in range(int(max_iter)):
            loss, dp = pdp_solver.step(ini_state, horizon, current_parameter)
            current_parameter -= lr * dp
        if return_obj:
            sol = pdp_solver.integrateSys(ini_state, horizon, current_parameter)
            obj_list.append(sol['cost'])
            state_list.append(sol['state_traj'][:-1, :])
            control_list.append(sol['control_traj'])
    end_time = time.time()
    time_cost = end_time - start_time

    if return_obj:
        obj_arr = np.array(obj_list).squeeze()
        control_arr, state_arr = np.array(control_list), np.array(state_list)
        return obj_arr, control_arr, state_arr, time_cost
    else:
        return time_cost


def pinn(system_name, target_list, return_obj=True):
    '''PINN with MLP backbone ''' 
    model_name = "PINN"
    config = Config()
    config.set_(model_name, system_name)
    # ---- env (dynamics and hemilton) ----
    def get_hamilton(config, s_target, x, lambda_, u):
        if config.system_name == "Pendulum":
            # constants
            l=1
            m=1
            damping_ratio=0.05
            g = 10
            ini_state = [0, 0]
            wq=10
            wdq=1
            loss_coef = torch.tensor([wq, wdq]).float()
            wu=0.1

            #dynamics (d(x, u) in Eq(2) of paper)
            d0 = x[:,1]
            d1 = (u[:,0] - m*g*l* torch.sin(x[:,0]) - damping_ratio* x[:,1]) / (m*l*l/3)
            d = torch.stack((d0, d1), dim=1)

            #cost (p(x, u) in Eq(2) of paper)
            s_target = torch.repeat_interleave(s_target, x.shape[0]//s_target.shape[0], dim=0)
            p = (x - s_target).pow(2).sum(dim=0).dot(loss_coef)
            u_penalty =  u.pow(2).sum() * wu 
            p = p + u_penalty

            #Hamilton
            h = p +  (lambda_ * d).sum()
        return h
    
    # ---- params ----
    _, ini_state, _, s_target_base = JinEnv.get_env(config.system_name)
    ini_state = torch.tensor(ini_state).float()

    horizon = config.n_sensor
    T = config.T + np.random.rand(1) * config.T / horizon  #random T for flexible time.
    dt = T / horizon
    t_ = np.arange(horizon) * dt 
    
    config.n_epoch = 100000
    batch_size = horizon
    obj_list = []
    control_list = []
    state_list = []
    start_time = time.time()
    obj_dm = [5582.82037357, 5246.83798176, 7162.34312089, 5529.79005478, 4081.76512671]
    for i, s_target_i in enumerate(target_list):
        s_target = torch.tensor(s_target_i).float()[None, :] #shape = (1, s_dim)
        t = torch.tensor(t_)[:,None].float().requires_grad_(True) #shape = (horizon, t_dim=1)
        
        # --- model and optim ---
        width = 40
        depth = 2
        mlp_width = [config.t_dim] + [width]*depth 
        x_model = ocm.MLP(mlp_width+[config.s_dim])
        lambda_model = ocm.MLP(mlp_width+[config.s_dim])
        u_model = ocm.MLP(mlp_width+[config.u_dim])
        lr = 0.01
        optimizer = optim.Adam(list(x_model.parameters()) + list(lambda_model.parameters()) + list(u_model.parameters()), lr=lr)
        scheduler = StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay_rate)
        
        for epoch in range(config.n_epoch + 1):
            #forward            
            x = x_model(t)
            lambda_ = lambda_model(t)
            u = u_model(t)
            #x.shape = (horizon, x_dim), lambda.shape=x.shape
            #u.shape = (horizon, u_dim)

            #time derivatives by autograd.grad  
            dx_dt = torch.cat(grad((x[:,i].sum() for i in range(config.s_dim)), (t for _ in range(config.s_dim)), create_graph=True), dim=1)  ##shape = (horizon, config.s_dim)
            dlambda_dt = torch.cat(grad((lambda_[:,i].sum() for i in range(config.s_dim)), (t for _ in range(config.s_dim)), create_graph=True), dim=1)
            
            #calculate hamilton
            h = get_hamilton(config, s_target, x, lambda_, u)  #shape= scalar

            # partial dirivatives of hamilton
            dh_dx = grad(h, x, create_graph=True)[0]
            dh_dlambda =  grad(h, lambda_, create_graph=True)[0]
            dh_du =  grad(h, u, create_graph=True)[0]
            

            # pinn loss
            dynamic_loss = (dx_dt - dh_dlambda).pow(2).sum() # Eq.3a
            costate_loss = (dlambda_dt + dh_dx).pow(2).sum() # Eq.3b
            oc_loss = (dh_du).pow(2).sum() # Eq.3c
            boundry_loss = (x[0]- ini_state).pow(2).sum() + (lambda_[-1]).pow(2).sum() #Eq.3d  #for n_prob>1, modify to x[:, 0, :],  lambda_[:, 0, :]
            pmp_loss = dynamic_loss + costate_loss + oc_loss
            pinn_loss = pmp_loss + boundry_loss
            pinn_loss = pinn_loss 

            # optimize
            pinn_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            config.n_print_epoch = 1000
            if epoch % config.n_print_epoch == 0:
                print("Epoch {}, pinn_loss:{:e}, dynamic_loss:{:e}, costate_loss:{:e}, oc_loss:{:e}, boundry_loss:{:e}".format(epoch, pinn_loss.item(), dynamic_loss.item(), costate_loss.item(), oc_loss.item(),  boundry_loss.item()), flush=1)

        if return_obj:
            u = u.detach().numpy()
            obj_i, control_i, state_i = eval_obj(u, config.system_name, s=None, goal=s_target_i)
            obj_list.append(obj_i)
            control_list.append(control_i)
            state_list.append(state_i)
            print(i, obj_i)
            if i < 5:
                print("err=", abs(obj_i-obj_dm[i])/obj_dm[i] )

    end_time = time.time()
    time_cost = end_time - start_time

    if return_obj:
        obj = torch.tensor(obj_list).squeeze().numpy()
        control = np.array(control_list)
        state = np.array(state_list)
        return obj, control, state, time_cost
    else:
        return time_cost


def twoPhasesControl(model_name, system_name, target_list, return_obj=True):
    # --- load model ---
    config = Config()
    config.set_(model_name, system_name)
    filename = "{}_{}_{}_{}.pth".format(model_name, system_name, config.n_sensor, config.n_train)
    model_path = os.path.join("./model", filename)
    model = ocm.get_model(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    # --- set lr and iter --- 
    if system_name == "Pendulum": 
        lr = 1e-2
        max_iter = 1000

    if system_name == "RobotArm":
        lr = 1e-2
        max_iter = 600
        
    elif system_name == "CartPole":
        lr = 1e-1
        max_iter = 1000

    elif system_name == "Quadrotor": #too bad, need re-train
        lr = 1e-3
        max_iter = 10000

    elif system_name == "Rocket":
        lr = 1e-2
        max_iter = 1000
    
    # --- set u and opter ---
    U = ocm.get_u(config)
    optimizer = optim.RMSprop(U.parameters(), lr=lr)

    # --- loss coef ---
    _, _, loss_weight, _ = JinEnv.get_env(system_name)
    loss_coef = torch.tensor(loss_weight[:-1]).float()
    u_coef = loss_weight[-1]
    if system_name in ["Quadrotor", "Rocket"]:
        loss_coef = loss_coef.repeat_interleave(3) # each 3 params is a group using same coef
    

    # --- infer u ---
    u_list = []
    t = np.linspace(0, config.T, num=config.n_sensor)[:, None] 
    t = torch.tensor(t).float().to(config.device)
    start_time = time.time()
    for target in target_list:
        y_target = torch.tensor(target).float()
        best_loss = 1e10
        best_param = None
        for i in range(max_iter):
            u_pred = U(t.squeeze()).flatten()  #output shape (n_sensor*u_dim,)
            u_tiled = torch.tile(u_pred[None,:], (config.n_sensor, 1)) #tiled shape (config.n_sensor, n_sensor*u_dim)
            y_pred = model(u_tiled, t) #shape (n_sample, s_dim)
            
            loss = (y_pred - y_target).pow(2).sum(dim=0).dot(loss_coef)
            u_penalty =  u_pred.pow(2).sum() * u_coef 
            loss = loss + u_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if return_obj and loss < best_loss:
                best_loss = loss
                best_param = U.state_dict()
            if i % 100 == 0:
                u = u_pred.detach().cpu().numpy().reshape((config.n_sensor, -1))
                obj, _, _ = eval_obj(u, system_name, s=None, goal=y_target)
                print("Epoch {}, obj: {:e}. loss: {:e} = {:e} + {:e}(u_penalty)".format(i, obj.item(), loss.item(), (loss-u_penalty).item(), u_penalty.item() ), flush=True)
        if return_obj:
            U.load_state_dict(best_param)
            u_pred = U(t.squeeze()).detach()
            u_list.append(u_pred)

    end_time = time.time()
    time_cost = end_time - start_time


    # --- calculte obj ---
    if return_obj:
        obj_list = []
        for i in range(len(target_list)):
            obj_i, _, _ = eval_obj(u_list[i], system_name, s=None, goal=target_list[i])
            obj_list.append(obj_i)
        obj = torch.tensor(obj_list).squeeze().cpu().numpy()
        return obj, time_cost
    else:
        return time_cost

def twoPhasesControl_param(model_name, system_name, target_list, return_obj=True):
    # --- load model ---
    config = Config()
    config.set_(model_name, system_name)
    filename = "{}_{}_{}_{}.pth".format(model_name, system_name, config.n_sensor, config.n_train)
    model_path = os.path.join("./model", filename)
    model = ocm.get_model(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    # --- set lr and iter --- 
    if system_name == "Pendulum": 
        lr = 1e-2
        max_iter = 1000

    if system_name == "RobotArm":
        lr = 1e-2
        max_iter = 600
        
    elif system_name == "CartPole":
        lr = 1e-1
        max_iter = 1000

    elif system_name == "Quadrotor": #too bad, need re-train
        lr = 1e-3
        max_iter = 10000

    elif system_name == "Rocket":
        lr = 1e-2
        max_iter = 1000
    
    # --- init u with a random train sample ---
    U = ocm.get_u(config)
    train_data = dg.get_data(config.model_name, config.system_name, config.T,  config.n_train, config.s_offset, config.s_range)
    X, _ = train_data[0]
    param = X[:-config.t_dim]
    U = ocm.get_u(config)
    U.linear.weight.data = param.reshape(U.linear.weight.shape).requires_grad_(True)

    # --- loss coef ---
    _, _, loss_weight, _ = JinEnv.get_env(system_name)
    loss_coef = torch.tensor(loss_weight[:-1]).float()
    u_coef = loss_weight[-1]
    if system_name in ["Quadrotor", "Rocket"]:
        loss_coef = loss_coef.repeat_interleave(3) # each 3 params is a group using same coef
    

    # --- infer u ---
    u_list = []
    t = np.linspace(0, config.T, num=config.n_sensor)[:, None] 
    t = torch.tensor(t).float().to(config.device)
    start_time = time.time()
    for target in target_list:
        optimizer = optim.RMSprop(U.parameters(), lr=lr) #new problem has new opter
        y_target = torch.tensor(target).float()
        best_loss = 1e10
        best_param = None
        for i in range(max_iter):
            u_pred = U(t.squeeze()).flatten()  #output shape (n_sensor*u_dim,)
            
            param = U.linear.weight.flatten()
            param_tiled = torch.tile(param[None,:], (config.n_sensor, 1)) #tiled shape (config.n_sensor, n_sensor*param_num)
            y_pred = model(param_tiled, t) #shape (n_sample, s_dim)
            
            loss = (y_pred - y_target).pow(2).sum(dim=0).dot(loss_coef)
            u_penalty =  u_pred.pow(2).sum() * u_coef 
            loss = loss + u_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if return_obj and loss < best_loss:
                best_loss = loss
                best_param = U.state_dict()
            if i % 100 == 0:
                u = u_pred.detach().cpu().numpy().reshape((config.n_sensor, -1))
                obj, _, _ = eval_obj(u, system_name, s=None, goal=y_target)
                print("Epoch {}, obj: {:e}. loss: {:e} = {:e} + {:e}(u_penalty)".format(i, obj.item(), loss.item(), (loss-u_penalty).item(), u_penalty.item() ), flush=True)
        if return_obj:
            U.load_state_dict(best_param)
            u_pred = U(t.squeeze()).detach()
            u_list.append(u_pred)

    end_time = time.time()
    time_cost = end_time - start_time


    # --- calculte obj ---
    if return_obj:
        obj_list = []
        for i in range(len(target_list)):
            obj_i, _, _ = eval_obj(u_list[i], system_name, s=None, goal=target_list[i])
            obj_list.append(obj_i)
        obj = torch.tensor(obj_list).squeeze().cpu().numpy()
        return obj, time_cost
    else:
        return time_cost

def execute_control(u, config, target_list):
    obj_list = []
    control_list = []
    state_list = []
    for i in range(len(target_list)):
        u_i = u[i*config.n_sensor: (i+1)*config.n_sensor]
        if config.t_dim > 1: # PDE
            u_i = u_i.reshape([int(config.T/config.dt)+1] * config.t_dim)
        
        item = target_list[i]
        goal, s0, dyn_args = arr2ocp(item, config)
        obj_i, control_i, state_i = eval_obj(u_i, config.system_name, s=None, goal=goal, s0=s0, dyn_args=dyn_args)
        obj_list.append(obj_i)
        control_list.append(control_i)
        state_list.append(state_i)
    obj = np.array(obj_list).squeeze()
    control = np.array(control_list)
    state = np.array(state_list)
    return obj, control, state

def load_model(config):
    if config.mode == 't': #target state:
        filename = "{}_{}_{}_{}.pth".format(config.model_name, config.system_name, config.n_sensor, config.n_train)    
    else:
        config.mode = ''.join(sorted(config.mode))
        filename = "{}_{}_{}_{}_{}.pth".format(config.model_name, config.system_name, config.mode, config.n_sensor, config.n_train)      
    if config.save_Np == True: #used for Np experiments.
        filename = "{}_{}_{}_{}_{}.pth".format(config.model_name, config.system_name, config.n_sensor, config.n_train, config.Np)    
    model_path = os.path.join("./model", filename)
    model = ocm.get_model(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    return model

def mlpControl(model_name, system_name, target_list, return_obj=True, n_train=None, mode='t', loaded_model=None):
    # --- set config ---
    config = Config()
    config.set_(model_name, system_name, mode)
    if not n_train is None:
        config.n_train = n_train
    
    # --- load model ---
    if loaded_model is None:
        model = load_model(config)
    else:
        model = loaded_model

    # --- prepare data ---
    t_list = []
    s_list = []
    for item in target_list:
        if config.t_dim == 1: #ODE
            t = np.linspace(0, config.T, num=config.n_sensor, endpoint=False)[:, None]
        else: #PDE
            interval = np.linspace(0, config.T, num=int(config.T/config.dt)+1).tolist()
            t = np.array(list(itertools.product(* [interval]*config.t_dim)))
        t_list.append(t)
        s = np.tile(item, (len(t), 1))
        s_list.append(s)
    t = np.concatenate(t_list, axis=0)
    t = torch.tensor(t).float().to(config.device)
    s = np.concatenate(s_list, axis=0)
    s = torch.tensor(s).float().to(config.device)

    # --- solve ---
    with torch.no_grad():
        start_time = time.time()
        u = model(s, t)
        end_time = time.time()
    time_cost = end_time - start_time

    # --- calculate obj ---
    u = u.cpu().numpy()
    if return_obj:
        obj, control, state = execute_control(u, config, target_list)
        return obj, control, state, time_cost
    else:
        return u, time_cost


def deepControl(model_name, system_name, target_list, return_obj=True, n_train=None, mode='t', loaded_model=None, new_dt=None):
    #print(target_list[:3])
    #raise ValueError
    # --- set config ---
    config = Config()
    config.set_(model_name, system_name, mode)
    if not new_dt is None:
        config.dt = new_dt
        config.n_sensor = int(config.T / config.dt)+1
    if not n_train is None:
        config.n_train = n_train
    
    # --- load model ---
    if loaded_model is None:
        model = load_model(config)
    else:
        mode = ''.join(sorted(mode))
        filename = "{}_{}_{}_{}_{}.pth".format(model_name, system_name, mode, config.n_sensor, config.n_train)    
    model_path = os.path.join("./model", filename)
    model = ocm.get_model(config)
    if return_obj:
        model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()


    # --- prepare data ---
    if config.t_dim == 1: #ODE
        t = np.linspace(0, config.T, num=config.n_sensor, endpoint=False)[:, None]
    else: #PDE
        interval = np.linspace(0, config.T, num=int(config.T/config.dt)+1).tolist()
        t = np.array(list(itertools.product(* [interval]*config.t_dim)))
    t = torch.tensor(t).float().to(config.device)
    s = np.array(target_list)
    s = torch.tensor(s).float().to(config.device)

    # --- solve ---
    with torch.no_grad():
        start_time = time.time()
        u = model.forward_oc_solver(s, t)
        end_time = time.time()
    time_cost = end_time - start_time

    # --- calculate obj ---
    u = u.cpu().numpy()
    if return_obj:
        obj, control, state = execute_control(u, config, target_list)
        return obj, control, state, time_cost
    else:
        return u, time_cost


def deepStochasticControl(model_name, system_name, target_list, return_obj=True, data_dist="in"):
    # --- load model ---
    config = Config()
    config.set_(model_name, system_name)
    filename = "{}_{}_{}_{}.pth".format(model_name, system_name, config.n_sensor, config.n_train)    
    model_path = os.path.join("./model", filename)
    model = ocm.get_model(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    # --- prepare data ---
    s = np.array(target_list)
    s = torch.tensor(s).float().to(config.device)

    # --- prepare envs ---
    env, env_args = JinEnv.get_env(system_name)
    
    # --- solve ---
    start_time = time.time()
    obj_list = []
    with torch.no_grad():
        for i, target_i in enumerate(target_list):
            seed = hash(data_dist + str(i)) % (2**32)  #fix the seed for each test probs
            env.seed(seed)
            tot_reward = 0
            obs = env.reset()
            for t in range(1000):
                action = model(s[i][None,:], torch.tensor(obs)[None,:]).numpy()
                action = np.array([action,])
                obs, rewards, dones, info = env.step(action)
                tot_reward += rewards
            obj = - tot_reward
            obj_list.append(obj)

        
    end_time = time.time()
    time_cost = end_time - start_time

    return np.array(obj_list), time_cost
    


def get_test_sets(system_name, mode, n_test, n_times_s_out_offset=1):
    config = Config()
    config.set_(model_name=None, system_name=system_name, mode= mode)
    data_split = {"in":[config.s_offset, config.s_range], "out":[config.s_out_offset * n_times_s_out_offset, config.s_out_range]}
    test_sets = []
    for split, info in data_split.items():
        if mode == 't':
            if n_times_s_out_offset == 1:
                file_path = "./data/DeepControl_{}_{}_test_{}.npz".format(system_name, split, n_test)
            else:
                file_path = "./data/DeepControl_{}_{}_{}_test_{}.npz".format(system_name, split, n_times_s_out_offset, n_test)
        else:
            mode = ''.join(sorted(mode))
            if n_times_s_out_offset == 1:
                file_path = "./data/DeepControl_{}_{}_{}_test_{}.npz".format(system_name, mode, split, n_test)
            else:
                file_path = "./data/DeepControl_{}_{}_{}_{}_test_{}.npz".format(system_name, mode, split, n_times_s_out_offset, n_test)
            
        if os.path.exists(file_path):
            data = np.load(file_path)
            X = data["X"]
        else:
            print("Generating test sets..")
            generator = dg.OC_Data_Generator(config.T, system_name)
            n_sensor = config.n_sensor
            X, y = generator.gen_data(n_test * n_sensor, s_offset=info[0], s_range=info[1], mode=mode, n_sample_per_trial=n_sensor)
            X = X[0]  # s_target, shape = (n_test * n_sensor, s_dim)
            X = X[np.arange(0, n_test * n_sensor, n_sensor) ] # remove repetation, shape=(n_test, s_dim)
            np.savez_compressed(file_path, X=X)
        test_sets.append(X) #shape=(n_test, s_dim)
    

    in_X, out_X = test_sets
    
    return list(in_X), list(out_X)

def get_stochastic_test_sets(system_name, n_test):
    config = Config()
    config.set_(model_name=None, system_name=system_name)

    #in test 
    file_path = "./data/DeepControl_{}_in_test_{}.npz".format(system_name, n_test)
    if os.path.exists(file_path):
        data = np.load(file_path)
        in_target, in_reward = data["X"], data['y']
    else:
        generator = dg.StoOC_Data_Generator()
        in_target, in_reward = generator.gen_test_data(n_test, config.s_offset, config.s_range, data_dist="in")
        np.savez_compressed(file_path, X=in_target, y=in_reward)

    
    #out test
    file_path = "./data/DeepControl_{}_out_test_{}.npz".format(system_name, n_test)
    if os.path.exists(file_path):
        data = np.load(file_path)
        out_target, out_reward = data["X"], data['y']
    else:
        generator = dg.StoOC_Data_Generator()
        out_target, out_reward = generator.gen_test_data(n_test, config.s_out_offset, config.s_out_range, data_dist="out")
        np.savez_compressed(file_path, X=out_target, y=out_reward)
    

    
    in_cost = - in_reward
    out_cost = - out_reward
    return in_target.tolist(), in_cost, out_target.tolist(), out_cost
    
def get_push_test_sets(system_name, n_test):
    config = Config()
    config.set_(model_name=None, system_name=system_name)
    data_split = {"in":[config.s_offset, config.s_range], "out":[config.s_out_offset, config.s_out_range]}
    test_sets = []
    for split, info in data_split.items():
        file_path = "./data/DeepControl_{}_{}_test_{}_20230323.npz".format(system_name, split, n_test)
        if os.path.exists(file_path):
            data = np.load(file_path)
            X, y = data["X"], data['y']
        else:
            if system_name == "Pushing":
                generator = dg.Push_All_Data_Generator(config.T, system_name)
            elif system_name.startswith("Pushing_"):
                generator = dg.Push_Data_Generator(config.T, system_name)
            n_sensor = config.n_sensor
            X, y = generator.gen_data(n_test * n_sensor, info[0], info[1], n_sample_per_trial=n_sensor, sampling=False)
            X = X[0]  # s_target, shape = (n_test * n_sensor, s_dim)
            
            #print(n_test)
            #print(n_sensor)
            X = X[np.arange(0, n_test * n_sensor, n_sensor) ] # remove repetation, shape=(n_test, s_dim)
            y = y.reshape((n_test, n_sensor))  # shape (n_test*n_sensor) --> (n_test, n_sensor )  (u_dim=1 omittted)
            np.savez_compressed(file_path, X=X, y=y)
        test_sets.append(X) #shape=(n_test, s_dim)
        test_sets.append(y) # shape=(n_test, n_sensor)  (u_dim=1 omittted)

    in_X, in_y, out_X, out_y = test_sets
    
    return in_X, in_y, out_X, out_y


def get_AnaOCP_test_sets(system_name, n_test):
    """ OCP systems with analytical solutions, including Brachistochrone and Zermelo.
    """
    config = Config()
    config.set_(model_name=None, system_name=system_name)
    data_split = {"in":[config.s_offset, config.s_range], "out":[config.s_out_offset, config.s_out_range]}
    test_sets = []
    for split, info in data_split.items():
        file_path = "./data/DeepControl_{}_{}_test_{}.npz".format(system_name, split, n_test)
        
        if os.path.exists(file_path):
            data = np.load(file_path)
            X, y_dm, y_ana = data["X"], data['y_dm'], data['y_ana']
        else:
            if system_name == "Brachistochrone":
                generator = dg.Brachistochrone_Data_Generator(config.T, system_name)
            elif system_name == "Zermelo":
                generator = dg.Zermelo_Data_Generator(config.T, system_name)
            n_sensor = config.n_sensor
            X, y_dm = generator.gen_data(n_test * n_sensor, info[0], info[1], n_sample_per_trial=n_sensor, sampling=False)
            X = X[0]  # s_target, shape = (n_test * n_sensor, s_dim)
            X = X[np.arange(0, n_test * n_sensor, n_sensor) ] # remove repetation, shape=(n_test, s_dim)
            y_dm = y_dm.reshape((n_test, n_sensor))  # shape (n_test*n_sensor) --> (n_test, n_sensor )  (u_dim=1 omittted)
            #analytical solver
            y_ana = np.zeros((n_test, n_sensor))
            if system_name == "Brachistochrone":
                sys = PDP.BrachistochroneSys()
                for i in range(n_test):
                    _, k, Theta = sys.get_instance(ini_state=X[i], x_bound=config.T)
                    y_ana[i,:] = sys.ana_solver(ini_state=X[i], horizon=config.n_sensor-1, dx=config.T/(config.n_sensor-1), k=k, Theta=Theta)
            elif system_name == "Zermelo":
                sys = PDP.ZermeloSys()
                for i in range(n_test):
                    y_ana[i,:] = sys.ana_solver(ini_state=X[i], horizon=config.n_sensor-1)
            np.savez_compressed(file_path, X=X, y_dm=y_dm, y_ana=y_ana)
        test_sets.append(X) #shape=(n_test, s_dim)
        test_sets.append(y_dm) # shape=(n_test, n_sensor)  (u_dim=1 omittted)
        test_sets.append(y_ana)
    
    
    return test_sets #[in_X, in_y_dm, in_y_ana, out_X, out_y_dm, out_y_ana]

def benchmark_obj(model_name, system_name, mode='t', print_res=True, loaded_model=None):
    if print_res:
        print("\n\n ------- {}, {} -------".format(model_name, system_name), flush=True)
    
    config = Config()
    config.set_(model_name, system_name, mode)
    n_test = config.n_benchmark


    if system_name.startswith("Pushing"):
        in_target, ref_in_obj, out_target, ref_out_obj = get_push_test_sets(system_name, n_test)
        # here ref_in_obj, ref_out_obj are ground truth u trajectory, with shape (n_test, horizon, u_dim)
        target = np.concatenate((in_target, out_target), axis=0)
        #if model_name in ["MLPControl-dp", "MLPControl", "FNOControl"] or model_name in ["CNOControl","CNOControl1","CNOControl2"] or model_name.startswith("NSMControl"):  # Other Neural Operators
        
        if model_name.startswith("DeepControlNet") or model_name in ["GENControl"]:  # Accelerated Eval
            u, tot_time =deepControl(model_name, system_name, target, return_obj=False, loaded_model=loaded_model)
        else:
            u, tot_time =mlpControl(model_name, system_name, target, return_obj=False, loaded_model=loaded_model)
        
        _, horizon = ref_in_obj.shape
        
        u = u.reshape((-1, horizon)) #shape (2*n_test*horizon) --> (2*n_test, horizon)
        in_obj, out_obj = u[:len(in_target)], u[len(in_target):]  #for consistency, here we name u as 'obj' 
    
    elif system_name in ["Brachistochrone", "Zermelo"]:
        in_X, in_y_dm, in_y_ana, out_X, out_y_dm, out_y_ana = get_AnaOCP_test_sets(system_name, n_test)
        
        target = np.concatenate((in_X, out_X), axis=0)
        #if model_name in ["MLPControl", "FNOControl", "CNOControl2t"] or model_name.startswith("NSMControl"):  # Other Neural Operators
        if model_name.startswith("DeepControlNet") or model_name in ["GENControl"]: # Accelerated Eval
            u, tot_time =deepControl(model_name, system_name, target, return_obj=False, loaded_model=loaded_model)
        elif "Control" in model_name: #neural models
            u, tot_time =mlpControl(model_name, system_name, target, return_obj=False, loaded_model=loaded_model)
        elif model_name == "DirectMethod":
            u = np.concatenate((in_y_dm, out_y_dm))
        else:
            raise NotImplementedError
        _, horizon = in_y_ana.shape
        u = u.reshape((-1, horizon)) #shape (n_test*horizon) --> (n_test, horizon)
        in_obj, out_obj = u[:len(in_X)], u[len(in_X):]  #for consistency, here we name u as 'obj' 
        ref_in_obj, ref_out_obj = in_y_ana, out_y_ana
        #print(in_obj[99])
        #print(ref_in_obj[99])

    else:
        in_target_list, out_target_list = get_test_sets(system_name, mode, n_test)
        
        
        ref_in_obj, _, _, _ = direct_method(system_name, in_target_list, mode=mode)
        ref_out_obj, _, _, _ = direct_method(system_name, out_target_list, mode=mode)


        if model_name == "PDP":
            in_obj, _, _, _ = pdp(system_name, in_target_list, mode=mode)
            out_obj, _, _, _ =  pdp(system_name, out_target_list, mode=mode)
            
        
        elif model_name.startswith("DeepControlNet") or model_name in ["GENControl"] :  # Accelerated Eval
            obj, _, _, _ = deepControl(model_name, system_name, in_target_list+out_target_list, mode=mode, loaded_model=loaded_model)
            in_obj, out_obj = obj[:len(in_target_list)], obj[len(in_target_list):]

        #elif model_name in ["MLPControl-dp", "MLPControl", "FNOControl"] or model_name.startswith("CNOControl") or model_name.startswith("NSMControl"):  #  Neural Operators
        else:  #all other neural models
            obj, _, _, _ = mlpControl(model_name, system_name, in_target_list+out_target_list, mode=mode, loaded_model=loaded_model)
            in_obj, out_obj = obj[:len(in_target_list)], obj[len(in_target_list):]

        #


    if system_name.startswith("Pushing") or system_name in ["Brachistochrone"]:
        in_obj_err = np.abs(ref_in_obj - in_obj).sum(axis=(1,)) / np.abs(ref_in_obj).sum(axis=(1,))
        out_obj_err = np.abs(ref_out_obj - out_obj).sum(axis=(1,)) / np.abs(ref_out_obj).sum(axis=(1,))

    else:
        in_obj_err = np.abs(ref_in_obj - in_obj) / np.abs(ref_in_obj)
        out_obj_err = np.abs(ref_out_obj - out_obj) / np.abs(ref_out_obj)
    #print(in_obj_err)

    in_obj_err = reject_outliers(in_obj_err)
    out_obj_err = reject_outliers(out_obj_err)
    in_obj_err_bs = bootstrap(in_obj_err)
    out_obj_err_bs = bootstrap(out_obj_err) #std estimated by boostrap is less than np.std.  
    #print(in_obj_err_bs, out_obj_err_bs)

    #print(in_obj_err)

    if print_res:
        print(model_name, "In-test error avg={}, std={}".format(*bootstrap(in_obj_err) ))
        print(model_name, "Out-test error avg={}, std={}".format(*bootstrap(out_obj_err) ))

        
    #return np.mean(in_obj_err), np.std(in_obj_err), np.mean(out_obj_err), np.std(out_obj_err)
    return in_obj_err_bs[0], in_obj_err_bs[1], out_obj_err_bs[0], out_obj_err_bs[1]


def finetune(model, config, n_times_s_out_offset):
    import train as tr
    # --- load data ---
    out_tune_data = dg.get_data(config.model_name, config.system_name, config.T, config.n_test, config.s_out_offset * n_times_s_out_offset, config.s_out_range, config.mode)
    test_loader_args = {'batch_size': config.batch_size_test,  'shuffle': True, 'pin_memory':config.device=="cuda", 'num_workers':0}
    out_tune_loader = torch.utils.data.DataLoader(out_tune_data, **test_loader_args)
    
    # --- train ---
    n_epoch = config.n_epoch //5
    optimizer = optim.Adam(model.parameters(), lr=config.lr/10)
    if config.model_name == "CNOControl2t":
        model.branch_fc0.requires_grad = False
        model.branch0.requires_grad = False
        model.branch1.requires_grad = False
        model.trunk.requires_grad = False
    for epoch in range(n_epoch+1):
        loss = tr.train(model, device='cpu', train_loader=out_tune_loader, optimizer=optimizer)    
        #print("Epoch{}, loss={}".format(epoch, loss), flush=True)
    return model

def benchmark_obj_n_ood_shift(model_name, system_name, do_finetune=True, mode='t', print_res=True):
    '''Test ood obj with different s_out_offset and finetune'''
    if print_res:
        print("\n ------- {}, {}, finetune={} -------".format(model_name, system_name, do_finetune), flush=True)
        #print("OOD error of different level of shift")
    
    config = Config()
    config.set_(model_name, system_name, mode)
    n_test = config.n_benchmark
    if model_name == "PDP":
        n_test = 1

    #for n_times_s_out_offset in range(1,42,10):
    for n_times_s_out_offset in range(1,11,3):
        _, out_target_list = get_test_sets(system_name, mode, n_test, n_times_s_out_offset)

        ref_out_obj, _, _, _ = direct_method(system_name, out_target_list, mode=mode)

        if do_finetune:
            model = load_model(config)
            #model = ocm.get_model(config)
            model = finetune(model, config, n_times_s_out_offset)
        else:
            model = None

        if model_name == "PDP":
            out_obj, _, _, _ =  pdp(system_name, out_target_list, mode=mode)
                        
        elif model_name.startswith("DeepControlNet") or model_name in ["GENControl"] :  # Accelerated Eval
            out_obj, _, _, _ = deepControl(model_name, system_name, out_target_list, mode=mode, loaded_model=model)
        else:
            out_obj, _, _, _ = mlpControl(model_name, system_name, out_target_list, mode=mode, loaded_model=model)

        out_obj_err = np.abs(ref_out_obj - out_obj) / np.abs(ref_out_obj)
        #if n_test > 1:
        #    out_obj_err = reject_outliers(out_obj_err)

        if print_res:
            
            print(np.mean(out_obj_err) , end=',', flush=True)
            
    print(" ")
        
    


def benchmark_obj_n_train(system_name, n_test = 100):
    in_target_list, out_target_list = get_test_sets(system_name, mode='t', n_test=n_test)
    
    ref_in_obj, _, _, _ = direct_method(system_name, in_target_list)
    ref_out_obj, _, _, _ = direct_method(system_name, out_target_list)
    
    #data = {"n_train":[], "MLP_in_test":[[],[]], "DCN_in_test":[[],[]], "MLP_out_test":[[],[]], "DCN_out_test":[[],[]]}
    data = {'n_train':[], "OptCtrlOP_in":[], "OptCtrlOP_out":[]}
    for model_name in ["CNOControl2t"]:
        config = Config()
        config.set_(model_name, system_name)
        data['n_train'] = [1000, 2500, 5000, 10000, 20000, 40000, 80000, 100000]
        for n_train in data['n_train']:
            if n_train == 2000:
                n_train = 2001 #dataset bug patch
            
            if model_name == "MLPControl":
                abbr_name = "MLP"
            elif model_name == "CNOControl2t":
                abbr_name = "OptCtrlOP"
            
            obj, _, _, _ = mlpControl(model_name, system_name, in_target_list+out_target_list, n_train = n_train)
            in_obj, out_obj = obj[:len(in_target_list)], obj[len(in_target_list):]


            in_obj_err = reject_outliers( np.abs(ref_in_obj - in_obj) / np.abs(ref_in_obj))
            out_obj_err = reject_outliers( np.abs(ref_out_obj - out_obj) / np.abs(ref_out_obj))

            data[abbr_name+"_in"].append(np.mean(in_obj_err))
            data[abbr_name+"_out"].append(np.mean(out_obj_err))
            #data[abbr_name+"_in_test"][0].append(np.mean(in_obj_err))
            #data[abbr_name+"_in_test"][1].append(np.std(in_obj_err))
            #data[abbr_name+"_out_test"][0].append(np.mean(out_obj_err))
            #data[abbr_name+"_out_test"][1].append(np.std(out_obj_err))
    print(system_name, data)
    return data


        
    
def benchmark_stochastic_time(model_name, system_name, n_test=2000, n_trial = 10):
    
    # --- load model ---
    config = Config()
    config.set_(model_name, system_name)
    filename = "{}_{}_{}_{}.pth".format(model_name, system_name, config.n_sensor, config.n_train)    
    model_path = os.path.join("./model", filename)
    model = ocm.get_model(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    # --- prepare envs ---
    _, env_args = JinEnv.get_env(system_name)
    env = make_vec_env(JinEnv.StochasticPendulumEnv, n_envs=n_test, env_kwargs=env_args)

    # --- prepare data ---
    s = np.random.rand(n_test, len(env_args['x_goal']))
    s = torch.tensor(s).float().to(config.device)
    
    # --- solve ---
    model_time_list = []
    env_time_list = []
    # need to eval model time and env time.
    for i in range(n_trial):
        start_time = time.time()
        model_time = 0 
        env_time = 0
        with torch.no_grad():
            obs = env.reset()
            for t in range(1000):
                model_start_time = time.time()
                action = model(s, torch.tensor(obs))
                model_time += time.time()- model_start_time

                env_start_time = time.time()
                action = action.numpy()[:,None]
                obs, rewards, dones, info = env.step(action)
                #obs = np.random.rand(*obs.shape).astype(np.float32)
                env_time += time.time() - env_start_time
        model_time_list.append(model_time)
        env_time_list.append(env_time)
        


    model_time = np.array(model_time_list) / n_test
    env_time = np.array(env_time_list)

    print("{} time per prob: avg = {:e}, std= {:e}".format(model_name, np.mean(model_time), np.std(model_time)))
    print("total env time:   avg = {:e}, std= {:e}".format(np.mean(env_time), np.std(env_time)))

    #MLPControl time per prob: avg = 2.099701e-04, std= 1.872208e-06
    #DeepControlNet5 time per prob: avg = 3.329779e-04, std= 4.082740e-06
    #PPO time per prob: avg = 3.819233e01, std = 1.861774e-01
    # 50000 steps in 4-VecEnv time avg= 13.25, std = 0.12725

    # ------- DeepControlNet5, StochasticPendulum -------
    #DeepControlNet5 In-test error avg=0.0918805293847349, std=0.04592222731047207
    #DeepControlNet5 Out-test error avg=0.030446182903710317, std=0.01092740331337101

    #------- MLPControl, StochasticPendulum -------
    #MLPControl In-test error avg=0.09133354206547409, std=0.04588139805624502
    #MLPControl Out-test error avg=0.04690667263494883, std=0.010725250040073958

def benchmark_time(model_name, system_name, n_test = 2000, n_trial = 10, mode='t', new_dt=None):
    print("\n ------- {}, {} -------".format(model_name, system_name), flush=True)
    config = Config()
    config.set_(model_name, system_name, mode)
    

    
    if system_name == "Pushing":
        shape_dim = 11
        input_dim = config.s_dim - shape_dim + config.img_size[0] *  config.img_size[1]
        base_target = np.ones((input_dim,))
    elif system_name.startswith("Pushing"):
        base_target = np.ones((config.s_dim,))
    elif system_name.startswith("Brachistochrone"):
        base_target = np.array([2, 1])
    elif system_name.startswith("Zermelo"):
        base_target = np.array([1, 1, 3, 0, 0, 0, 0])
       
    else:
        _, _, _, base_target, _ = JinEnv.get_env(system_name)
        if mode == 't':
            base_target = np.array(base_target)
        else:
            base_target = np.zeros((config.input_dim,))
    time_cost_list = []
    np.random.seed(0) # fix seed for fixed test set

    for trial in range(n_trial):
        in_target_list, out_target_list = [], []
        for i in range(n_test):
            in_target = base_target + np.random.rand(len(base_target))* config.s_range + config.s_offset 
            out_target =  base_target + np.random.rand(len(base_target))* config.s_out_range + config.s_out_offset 
            in_target_list.append(in_target)
            out_target_list.append(out_target)
        target_list = in_target_list + out_target_list
        if model_name == "PDP":
            time_cost = pdp(system_name, target_list, return_obj=False, mode=mode)
        elif model_name == "PINN":
            time_cost = pinn(system_name, target_list, return_obj=False)
        elif model_name == "DirectMethod":
            time_cost = direct_method(system_name, target_list, return_obj=False, mode=mode, new_dt=new_dt)
        else:  # Other Neural Operators
            _, time_cost = mlpControl(model_name, system_name, target_list, return_obj=False, mode=mode)
        
        time_cost_list.append(time_cost)
    
    time_cost = np.array(time_cost_list) / len(target_list)
    if n_trial > 1:
        time_cost = reject_outliers(time_cost)
        print("time cost(s/prob) avg = {:e},  std={:e}".format(np.mean(time_cost), np.std(time_cost)), flush=True)
        return np.mean(time_cost), np.std(time_cost)
    else:
        print("time cost(s/prob) avg = {:e}".format(np.mean(time_cost)), flush=True)
        return np.mean(time_cost), 0
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def bootstrap(data, n_resamples=1000):
    replications = np.array([np.random.choice(data, len(data), replace = True) for _ in range(n_resamples)])
    result = np.mean(replications, axis=1)
    return np.mean(result), np.std(result)

def benchmark_obj_all_systems(model_name, mode='t'):
    result = []
    for system in ["Pendulum", "RobotArm", "CartPole", "Quadrotor", "Rocket"]:
        res = benchmark_obj(model_name, system, print_res=False, mode=mode)
        result.append(res)
    print("--------Obj of {} -------".format(model_name))
    print("4 rows, denoting in_avg, in_std, out_avg, out_std. 5 columns for 5 systems")
    print(np.array(result).T.tolist())

def benchmark_print_obj_all_systems_models():
    for dist in ["ID","OOD"]:
        print("-------- {} Obj -------".format(dist))
        for model_name in  ["DM", "Ours", "DON", "MLP", "GEN", "FNO"]:
            result = []
            for system_name in ["Pendulum",  "RobotArm", "CartPole", "Quadrotor", "Rocket", "Pushing", "Quadrotor-dit", "Brachistochrone", "Zermelo"]:
                _model_name = model_name
                if model_name == "Ours":
                    if system_name == "Pushing":
                        _model_name = "NSMControl9"
                    elif system_name in ["Brachistochrone", "Zermelo"]:
                        _model_name = "NSMControl6_2"
                    else:
                        _model_name = "NSMControl6"
                elif model_name == "DM":
                    _model_name = "DirectMethod"
                elif model_name == "DON":
                    _model_name = "DeepControlNet5"
                elif model_name in ["MLP", "GEN", "FNO"]:
                    _model_name = model_name + "Control"
                
                if model_name == "DM" and system_name in ["Pendulum",  "RobotArm", "CartPole", "Quadrotor", "Rocket", "Pushing", "Quadrotor-dit"]:
                    res = [0, 0]
                else:
                    _system_name = system_name
                    mode = 't'
                    if system_name == "Quadrotor-dit":
                        _system_name = "Quadrotor"
                        mode = "dit"

                    res = benchmark_obj(_model_name, _system_name, print_res=False, mode=mode)
                    #res = np.random.rand(4)
                    if dist == "ID":
                        res = (res[0], res[1])
                    else:
                        res = (res[2], res[3])
                result.append(res)
            print('"{}":'.format(model_name), np.array(result).T.tolist(), ',', flush=1)

def benchmark_time_all_systems(model_name):
    result = []
    for system_name in ["Pendulum",  "RobotArm", "CartPole", "Quadrotor", "Rocket", "Pushing", "Quadrotor-dit", "Brachistochrone", "Zermelo"]:
        _model_name = model_name
        if model_name.startswith("NSMControl"):
            if system_name == "Pushing":
                _model_name = "NSMControl9"
            elif system_name in ["Brachistochrone", "Zermelo"]:
                _model_name = "NSMControl6_2"
            else:
                _model_name = "NSMControl6"
        _system_name = system_name
        mode = 't'
        if system_name == "Quadrotor-dit":
            _system_name = "Quadrotor"
            mode = "dit"

        res = benchmark_time(_model_name, _system_name, mode=mode)
        result.append(res)

    print("--------Time of {} -------".format(model_name))
    print("2 rows, denoting avg and std. 9 columns for 9 systems")
    print(np.array(result).T.tolist())




if __name__ == "__main__":
    pass
    #utils.redirect_log_file()
    #models = ["DeepControlNet5", "MLPControl", "FNOControl", "GENControl"]
    #for model_name in models:
    #    benchmark_obj(model_name, "Quadrotor", mode="dit")
    #benchmark_time("PDP", "Quadrotor", mode="dit", n_test = 1, n_trial = 3)
    #systems=["Pendulum", "RobotArm", "CartPole", "Quadrotor", "Rocket"]


    