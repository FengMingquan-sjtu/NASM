import warnings
#warnings.filterwarnings("ignore")
import itertools
import os 
import argparse
import copy
import time 


import numpy as np
import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F



from config import Config
import data_generator as dg
import oc_model as ocm
import utils
from utils import timing
from evaluate import benchmark_obj



def train(model, device, train_loader, optimizer):
    # training a general deepoc model
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)

def train_with_reg(epoch, model, device, train_loader, optimizer, config):
    '''training regularized NSM model'''
    # generate gird points for regularization
    s_in_grid, t_grid = dg.get_reg_grid(config.system_name, config.T, config.n_train, config.s_offset, config.s_range, config.mode)
    s_out_grid, _     = dg.get_reg_grid(config.system_name, config.T, config.n_test,  config.s_out_offset, config.s_out_range, config.mode)
    s_grid = torch.cat((s_in_grid, s_out_grid), dim=0) #shape= (n_train+n_test, s_dim)
    s_grid, t_grid = s_grid.to(device), t_grid.to(device)

    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        supervise_loss = F.mse_loss(output, target)
        reg_loss, reg_loss_list = model.regularization(s_grid, t_grid)
        while reg_loss > supervise_loss:
            reg_loss /= 10.0
        loss = supervise_loss + reg_loss

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if epoch > config.prune_epoch_ratio * config.n_epoch and epoch % config.n_print_epoch == 0: #and torch.all(model.pruning_mask): #if not pruned
        model.prune()
        #after pruning, the sparsity is not required.
        model.coef_sparse= 0.0

    reg_loss = reg_loss.item()
    reg_loss_list = [l.item() for l in reg_loss_list]
    return train_loss / len(train_loader), reg_loss,  reg_loss_list



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target).item() 
    
    return test_loss / len(test_loader)

@timing
def main(config):
    
    ## get data
    train_data = dg.get_data(config.model_name, config.system_name, config.T,  config.n_train, config.s_offset, config.s_range, config.mode)
    test_data = dg.get_data(config.model_name, config.system_name, config.T, config.n_test, config.s_offset, config.s_range, config.mode)
    train_loader_args = {'batch_size': config.batch_size_train , 'shuffle': True, 'pin_memory':config.device.startswith("cuda"), 'num_workers':0}
    test_loader_args = {'batch_size': config.batch_size_test,  'shuffle': False, 'pin_memory':config.device.startswith("cuda"), 'num_workers':0}
    train_loader = torch.utils.data.DataLoader(train_data, **train_loader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **test_loader_args)    
    out_test_data = dg.get_data(config.model_name, config.system_name, config.T, config.n_test, config.s_out_offset, config.s_out_range, config.mode)
    out_test_loader = torch.utils.data.DataLoader(out_test_data, **test_loader_args)
    #get network
    device = torch.device(config.device)
    print("using device:", device, flush=True)
    model = ocm.get_model(config)


    #model save path
    print("Model {} with {} params".format(model.name, sum(p.numel() for p in model.parameters())))
    model_path = "./model"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    if config.mode == 't':
        name = "{}_{}_{}_{}".format(config.model_name, config.system_name, config.n_sensor, config.n_train)
    else:
        name = "{}_{}_{}_{}_{}".format(config.model_name, config.system_name, config.mode, config.n_sensor, config.n_train)
    if config.save_Np == True: #used for Np experiments.
        name = "{}_{}_{}_{}_{}".format(config.model_name, config.system_name, config.n_sensor, config.n_train, config.Np)    
    model_path = os.path.join(model_path, name+".pth")

    

    #optimization
    print("start train:",name, flush=True)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay_rate)
    best_test_loss = 1e100
    best_state_dict = None
    do_eval_obj = True
    time_limit = 3600 * 6  #6 hours
    start_time = time.time()

    for epoch in range(config.n_epoch+1):

        train(model, device, train_loader, optimizer)
        scheduler.step()
        if epoch % config.n_print_epoch == 0:
            train_loss  = test(model, device, train_loader)
            test_loss  = test(model, device, test_loader)
            out_test_loss  = test(model, device, out_test_loader)

            if do_eval_obj:
                err = benchmark_obj(config.model_name, config.system_name, config.mode, print_res=False, loaded_model=model)
                print("Epoch {}, train loss: {:e}, test loss: {:e}, out_test loss: {:e}, in_err_avg: {:e}, in_err_std: {:e}, out_err_avg: {:e}, out_err_std: {:e}".format(epoch, train_loss, test_loss, out_test_loss, err[0], err[1], err[2], err[3]), flush=True)
            else:
                print("Epoch {}, train loss: {:e}, test loss: {:e}, out_test loss: {:e}".format(epoch, train_loss, test_loss, out_test_loss), flush=True)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state_dict = copy.deepcopy(model.state_dict())
        time_cost = time.time() - start_time
        if time_cost > time_limit:
            break
    torch.save(best_state_dict, model_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
        help="which OC solver to use",
        choices=["NSMControl6", "DeepControlNet5", "MLPControl", "FNOControl", "GENControl"])
    parser.add_argument('--system_name', type=str, required=True,
        help="which OC system to solve",
        choices=["Pendulum", "RobotArm", "CartPole", "Quadrotor", "Rocket"]),
    parser.add_argument('--mode', type=str, default='t',
        help="input mode, t:target_state, i:ini_state, d:dynamics",)
    parser.add_argument('--print_log', action="store_true", 
        help="print/store the training info, if this flag is absent(default): store to log file, if this flag is present: print to terminal.")
    parser.add_argument('--log_root', type=str, required=False, default="./log",
        help="log file will be stored in which dir.")
    parser.add_argument('--n_train', type=int, required=False, default=0,
        help="Number of training samples, default 0 denotes using the pre-defined number.")
    parser.add_argument('--Np', type=int, required=False, default=0,
        help="Number of basis functions, default 0 denotes using the pre-defined number.")
    parser.add_argument('--device', type=str, required=False, default="cpu",
        help="training & testing device")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    config = Config()
    config.set_(args.model_name, args.system_name, args.mode)
    config.device = args.device
    if not args.print_log:
        utils.redirect_log_file(args.log_root)
    if args.n_train > 0:
        config.n_train =args.n_train 
    if args.Np >0:
        config.Np = args.Np
        config.save_Np = True

    main(config)


    


