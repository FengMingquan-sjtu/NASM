import os
import pickle
import math
import sys
import time
sys.path.append("./")

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from PDP import PDP, JinEnv
from  config import Config
from OC import test
import data_generator as dg
import oc_model as ocm
import utils


    
def debug_my_obj(name):
    env, ini_state, loss_weight, x_goal = JinEnv.get_env(name)
    wu = loss_weight[-1] #weight of penalty
    dt = config.dt
    horizon = int(config.T / dt)
    dyn = env.X + dt * env.f
    # --------------------------- create true OC solver ----------------------------------------
    true_solver = PDP.OCSys()
    true_solver.setStateVariable(env.X)
    true_solver.setControlVariable(env.U)
    true_solver.setDyn(dyn)
    true_solver.setPathCost(env.path_cost)
    true_sol = true_solver.ocSolver(ini_state=ini_state, horizon=horizon)
    true_state = true_sol['state_traj_opt'][:-1, :] # drop the last state for shape consistency. the output length is horizon.
    true_control = true_sol['control_traj_opt']
    true_obj = true_sol['cost']
    print("true_obj:", true_obj)
    print(true_control.shape)
    my_obj, my_control, my_state = eval_obj(true_control, name)
    
    print("my_obj", my_obj)
    # visualize
    fig_dir = "./figs/MyObj_{}".format(name)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    sensors = np.linspace(0, config.T, num=config.n_sensor)
    utils.draw_ys(sensors, {"MyObj":my_control ,"True":true_control}, "u", fig_dir)
    utils.draw_ys(sensors, {"MyObj":my_state ,"True":true_state}, "s", fig_dir)


def debug_don(name):
    system_name = name.split("_")[1]
    _, _, loss_weight, x_goal = JinEnv.get_env(system_name)
    device = torch.device("cpu")
    model_name = name.split("_")[0]
    model = ocm.get_model(model_name, device)
    model_path = os.path.join("./model", name+".pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for param in model.parameters(): #froze deeponet model
        param.requires_grad = False 
    
    t = np.linspace(0, config.T, num=config.n_sensor)[:, None] 
    t = torch.tensor(t).float().to(device)

    U = ocm.get_model(config.u_name, device)

    u_pred = U(t.squeeze()).flatten()  #output shape (n_sensor*u_dim, )
    _, _, s_true = eval_obj(u_pred, system_name)
    u_pred = torch.tile(u_pred[None,:], (config.n_sensor, 1)) #tiled shape (config.n_sensor, n_sensor*u_dim)
    s_pred = model(u_pred, t) #shape (n_sample, s_dim)

    # visualize
    fig_dir = os.path.join("./figs", "DebugDON_{}_{}".format(model_name, system_name))
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    utils.draw_ys(t, {"DON":s_pred, "True":s_true}, "DebugDON_{}_{}_s".format(model_name, system_name), fig_dir)

@utils.timing
def get_pdp_and_gt(name = "RobotArm"):
    """ pdp solution and true solution of oc solver.
    """
    
    # set store dir
    sol_path = "./solution"
    if not os.path.exists(sol_path):
        os.makedirs(sol_path)
    file_path = os.path.join(sol_path, "Control_{}_PDP.npz".format(name))
    
    if os.path.exists(file_path): # if solution is cached, read then return.
        data = np.load(file_path)
        pdp_state, pdp_control, true_state, true_control = data["pdp_state"], data["pdp_control"], data["true_state"], data["true_control"]
    
    else: # solve

        env, ini_state, loss_weight, x_goal = JinEnv.get_env(name)
        wu = loss_weight[-1] #weight of penalty
        dt = config.dt
        horizon = int(config.T / dt)
        dyn = env.X + dt * env.f
        # --------------------------- create true OC solver ----------------------------------------
        true_solver = PDP.OCSys()
        true_solver.setStateVariable(env.X)
        true_solver.setControlVariable(env.U)
        true_solver.setDyn(dyn)
        true_solver.setPathCost(env.path_cost)
        true_sol = true_solver.ocSolver(ini_state=ini_state, horizon=horizon)
        true_state = true_sol['state_traj_opt'][:-1, :] # drop the last state for shape consistency. the output length is horizon.
        true_control = true_sol['control_traj_opt']
        true_obj = true_sol['cost']
        print("true_obj:", true_obj)
        #print("my_obj:", eval_obj(true_control[:,0], name))
        #raise ValueError
        # --------------------------- create PDP Control/Planning object ----------------------------------------
        print("Control_{}_PDP".format(name))
        pdp_solver = PDP.ControlPlanning()
        pdp_solver.setStateVariable(env.X)
        pdp_solver.setControlVariable(env.U)
        pdp_solver.setDyn(dyn)
        pdp_solver.setPathCost(env.path_cost)
        # --------------------------- do the system control and planning ----------------------------------------
        # lr and n_iter of PDP
        if name == "RobotArm":
            lr = 1e-2
            max_iter = 5000
        elif name == "CartPole":
            lr = 1e-3
            max_iter = 5000
        elif name == "Pendulum":
            lr = 1e-4
            max_iter = 5000
        elif name == "Rocket":
            lr = 1e-4
            max_iter = 20000
        elif name == "Quadrotor":
            lr = 1e-4
            max_iter = 80000
        loss_trace, parameter_trace = [], []
        pdp_solver.init_step(horizon)
        initial_parameter = np.random.randn(pdp_solver.n_auxvar)
        current_parameter = initial_parameter
        
        for k in range(int(max_iter)):
            # one iteration of PDP
            loss, dp = pdp_solver.step(ini_state, horizon, current_parameter)
            # update
            current_parameter -= lr * dp
            loss_trace += [loss]
            parameter_trace += [current_parameter]
            # print
            if k % 100 == 0:
                u = pdp_solver.integrateSys(ini_state, horizon, current_parameter)['control_traj']
                u_penalty = wu * np.sum(u**2)
                print("Epoch {}, loss: {:e} = {:e} + {:e}(u_penalty)".format(k, loss, loss-u_penalty, u_penalty ))
                if k%1000 == 0:
                    print("", flush=True, end='')

        # solve the trajectory
        sol = pdp_solver.integrateSys(ini_state, horizon, current_parameter)
        pdp_state = sol['state_traj'][:-1, :] # drop the last state for shape consistency. the output length is horizon.
        pdp_control = sol['control_traj']
        print(name, "PDP params:", current_parameter)

        np.savez_compressed(file_path, pdp_state=pdp_state, pdp_control=pdp_control, true_state=true_state, true_control=true_control)

        # visualize
        fig_dir = "./figs/Control_{}_PDP".format(name)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        sensors = np.linspace(0, config.T, num=config.n_sensor)
        utils.draw_ys(sensors, { "PDP":pdp_control, "True":true_control}, "PDP_u", fig_dir)
        utils.draw_ys(sensors, { "PDP":pdp_state, "True":true_state}, "PDP_s", fig_dir)
    return pdp_state, pdp_control, true_state, true_control

@utils.timing
def get_two_phases_Control(name):
    """ 2 phases(DON + SGD) control baseline"""
    # set store dir
    sol_path = "./solution"
    if not os.path.exists(sol_path):
        os.makedirs(sol_path)
    file_path = os.path.join(sol_path, "Control_{}.npz".format(name))
    
    if os.path.exists(file_path): # if solution is cached, read then return.
        data = np.load(file_path)
        state, control = data['state'], data["control"]
        
    else:
        system_name = name.split("_")[1]
        _, _, loss_weight, x_goal = JinEnv.get_env(system_name)
        device = torch.device("cpu")
        model_name = name.split("_")[0]
        config = Config()
        config.set_(model_name, system_name)
        model = ocm.get_model(config)
        model_path = os.path.join("./model", name+".pth")
        device = config.device
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        
        #model.eval()
        #for param in model.parameters(): #froze deeponet model
        #    param.requires_grad = False 

        # ------------set storing dir ------------------------
        oc_name = "TwoPhasesControl"
        name = oc_name +"_"+name
        print("Setting:", name)
        fig_dir = os.path.join("./figs", name)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        
        # ------------ infer u  ------------------------
        t = np.linspace(0, config.T, num=config.n_sensor)[:, None] 
        t = torch.tensor(t).float().to(device)

        y_target = torch.tensor(x_goal).float()
        loss_coef = torch.tensor(loss_weight[:-1]).float()
        u_coef = loss_weight[-1]
        
        U = ocm.get_u(config)

        if system_name == "Pendulum": 
            lr = 1e-2
            max_iter = 800

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
        
        optimizer = optim.RMSprop(U.parameters(), lr=lr)

        if system_name in ["Quadrotor", "Rocket"]:
            loss_coef = loss_coef.repeat_interleave(3) # each 3 params is a group using same coef
        
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

            if loss < best_loss:
                best_loss = loss
                best_param = U.state_dict()
            if i % 100 == 0:
                u = u_pred.detach().cpu().numpy().reshape((config.n_sensor, -1))
                obj, _, _ = test.eval_obj(u, system_name, s=None, goal=y_target)
                print("Epoch {}, obj: {:e}. loss: {:e} = {:e} + {:e}(u_penalty)".format(i, obj.item(), loss.item(), (loss-u_penalty).item(), u_penalty.item() ), flush=True)
            

        # ------- visualize and comapre results --------------- 
        U.load_state_dict(best_param)
        u_pred = U(t.squeeze()).flatten()
        print("U params:", list(U.parameters()))
        control = u_pred.detach().cpu().numpy().reshape((config.n_sensor, -1))
        #state = y_pred.detach().cpu().numpy()
        #_,_, state = test.eval_obj(control, system_name, s=None, goal=y_target)
        print("result not saved")
        #np.savez_compressed(file_path, state=state, control=control)

    
        #pdp_state, pdp_control, true_state, true_control = get_pdp_and_gt(system_name)
        #utils.draw_ys(t.cpu().numpy(), {"TwoPhases":control, "PDP":pdp_control, "True":true_control}, "{}_{}_u".format(oc_name, system_name), fig_dir)
        #utils.draw_ys(t.cpu().numpy(), {"TwoPhases":state, "PDP":pdp_state, "True":true_state}, "{}_{}_s".format(oc_name, system_name), fig_dir)
        
    return state, control

def get_deepControl(name):
    system_name = name.split("_")[1]
    _, _, _, s_target = JinEnv.get_env(system_name)

    
    model_name = name.split("_")[0]
    config = Config()
    config.set_(model_name, system_name)
    model = ocm.get_model(config)
    model_path = os.path.join("./model", name+".pth")
    device = config.device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for param in model.parameters(): 
        param.requires_grad = False 

    # ------------set storing dir ------------------------
    oc_name = "Control_"
    name = oc_name +name
    print("Setting:", name)
    fig_dir = os.path.join("./figs", name)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    start_time = time.time()
    # ------------ infer u  ------------------------
    t = np.linspace(0, config.T, num=config.n_sensor)[:, None]
    t = torch.tensor(t).float().to(device)

    s = np.tile(s_target, (len(t), 1))
    s = torch.tensor(s).float().squeeze().to(device)

    u = model(s, t)

    # ------- timer, visualize and comapre results --------------- 
    end_time = time.time()
    time_cost = end_time - start_time
    if "DeepControlNet" in model_name:
        abbr_name = "DCN"
    elif "MLP" in model_name:
        abbr_name = "MLP"
    print("{} Infer u cost {}s.".format(abbr_name, time_cost))
    
    control = u.detach().cpu().numpy()
    _, pdp_control, _, true_control = get_pdp_and_gt(system_name)
    utils.draw_ys(t.cpu().numpy(), {abbr_name:control, "PDP":pdp_control, "True":true_control}, "{}{}_u".format(oc_name, system_name), fig_dir)



def benchmark_dcn(name, n_problem = 100):
    '''test the speed of dcn'''
    system_name = name.split("_")[1]
    _, _, _, s_target = JinEnv.get_env(system_name)
    device = torch.device("cpu")
    model_name = name.split("_")[0]
    model = ocm.get_model(model_name, device)
    model_path = os.path.join("./model", name+".pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for param in model.parameters(): #froze deeponet model
        param.requires_grad = False 
    
    # ------------ data prepare  ------------------------
    t_list = []
    s_list = []
    for i in range(n_problem):
        t = np.linspace(0, config.T, num=config.n_sensor)[:, None]
        t_list.append(t)
        s = np.tile(s_target + np.random.rand(len(s_target)), (len(t), 1))
        s_list.append(s)
    t = np.concatenate(t_list, axis=0)
    t = torch.tensor(t).float().to(device)
    s = np.concatenate(s_list, axis=0)
    s = torch.tensor(s).float().to(device)
        

    # ------------ infer u  ------------------------
    start_time = time.time()
    model(s, t)
    end_time = time.time()
    time_cost = end_time - start_time
    print("DCN infer u cost {}s.".format(time_cost))

def benchmark_direct_oc(name, n_problem = 100):
    '''test the speed of oc direct method, on Casadi, cpu'''
    env, ini_state, loss_weight, x_goal = JinEnv.get_env(name)
    wu = loss_weight[-1] #weight of penalty
    dt = config.dt
    horizon = int(config.T / dt)
    dyn = env.X + dt * env.f

    true_solver = PDP.OCSys()
    true_solver.setStateVariable(env.X)
    true_solver.setControlVariable(env.U)
    true_solver.setDyn(dyn)

    tot_time = 0
    for i in range(n_problem):
        s_target = x_goal + np.random.rand(len(x_goal))
        env.setTargetState(s_target)
        true_solver.setPathCost(env.path_cost)
        true_solver.setFinalCost(env.final_cost)

        start_time = time.time()
        true_sol = true_solver.ocSolver(ini_state=ini_state, horizon=horizon)
        end_time = time.time()
        tot_time += end_time - start_time
    
    print("DirectOC infer u cost {}s.".format(tot_time))

if __name__ == "__main__":
    #utils.redirect_log_file()
    #debug_my_obj("Quadrotor")
    #debug_don("DeepONet2_Quadrotor_100_100000")
    #get_pdp_and_gt(config.system_name)
    #get_deepControl("DeepControlNet_{}_100_20000".format(config.system_name))
    #get_deepControl("DeepControlNet2_{}_100_100000".format(config.system_name))
    #get_two_phases_Control("DeepONet2_Rocket_100_100000")
    #get_two_phases_Control("DeepONet2_Quadrotor_100_100000")
    #get_two_phases_Control("DeepONet2_Pendulum_GRF_100_10000")
    get_two_phases_Control("DeepONet2_RobotArm_GRF_100_10000")
    #get_deepControl("MLPControl_{}_100_100000".format(config.system_name))
    #benchmark_dcn("DeepControlNet_{}_100_20000".format(config.system_name))
    #benchmark_direct_oc(config.system_name)