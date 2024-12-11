import warnings
warnings.filterwarnings("ignore")
class Config:
    def __init__(self):
        '''Common params'''
        # --- device --- 
        self.device = "cpu"  #["cpu", "cuda"]

        # --- data_gen --- 
        self.batch_size_train = 10000
        self.batch_size_test = 10000
        self.s_offset = 0.1
        self.s_out_offset = -0.1 #out of distri s range and offset 
        self.s_out_range = 0.2
        self.img_size = [0, 0] #img input used in Pushing dataset

        self.n_sensor = 100#freqency of sampling t, 
        self.T = 1
        self.dt = 0.01 
        self.t_dim = 1

        # --- model --- 
        self.Np = None  #num of basis
        self.save_Np = False #weather include Np in model name
        
    
        # --- train --- 
        self.lr = 0.01
        self.lr_decay_rate = 0.9
        self.lr_decay_iter = 1000
        self.width = 40
    
        # --- eval & print ---
        self.n_benchmark = 100
        self.n_print = 20
        #self.n_print = 10
        
    def set_(self, model_name, system_name, mode='t'):
        '''Model and system specific params'''
        self.model_name = model_name  #["DeepONet*", "RON*", "DeepControlNet*", "MLPControl"]
        self.system_name = system_name  #["Pendulum", "RobotArm", "CartPole", "Quadrotor", "Rocket"]
        self.mode = ''.join(sorted(mode)) # a str of any combination of {'t', 'i', 'd'} denoting target, init, and dynamic
        if system_name == "Pendulum":
            self.s_dim = 2
            self.i_dim = 2
            self.d_dim = 3
            self.u_dim = 1
            self.depth = 7
            self.width = 24
            self.n_train, self.n_test = 5000, 2000
            self.s_range = 1
            self.n_epoch = 10000
            if model_name in ["DeepONet", "DeepONet2", "RON"]:
                self.n_train = 10000

        elif system_name == "RobotArm" :
            self.s_dim = 4
            self.i_dim = 4
            self.d_dim = 4
            self.u_dim = 1
            self.depth = 2
            self.n_train, self.n_test = 5000, 2000
            self.s_range = 1
            self.n_epoch = 10000

        elif system_name == "CartPole":
            self.s_dim = 4
            self.i_dim = 4
            self.d_dim = 3
            self.u_dim = 1
            self.depth = 3
            self.n_train, self.n_test = 5000, 2000
            self.s_range = 0.2
            self.n_epoch = 2000
            if model_name in ["DeepONet", "DeepONet2", "RON"]:
                self.depth = 2
                self.n_train = 10000
                self.n_epoch = 20000

        elif system_name == "Quadrotor":
            self.s_dim = 9
            self.i_dim = 13
            self.d_dim = 5
            self.u_dim = 4
            self.depth = 3
            self.n_train, self.n_test = 10000, 2000
            self.s_range = 1
            self.n_epoch = 500
            if self.mode == "dit":
                self.n_epoch = 1000

        elif system_name == "Rocket":
            self.s_dim = 9
            self.i_dim = 13
            self.d_dim = 5
            self.u_dim = 3
            self.depth = 3
            self.n_train, self.n_test = 10000, 2000
            self.s_range = 1
            self.n_epoch = 500
            if model_name in ["DeepONet", "DeepONet2", "RON"]:
                self.n_epoch = 1000

        elif system_name == "Pushing":
            # all shape and surfaces, with one-hot encoding
            self.s_dim = 3 + 2 +4 +2  +11  +22 # 44-dim(11dims exp setting, 11dims shape-encode, 22dims traj)
            self.img_size = [32, 24]
            self.u_dim = 1  #norm of force
            self.s_offset = 0.3
            self.s_range = 0.5
            self.s_out_offset = 0.21
            self.s_out_range = 0.09
            self.T = 0.44
            self.dt = 1/250
            self.n_sensor = 110

            self.depth = 3
            self.n_train, self.n_test = 50000, 2000
            self.n_epoch = 2000
            self.lr = 0.01

        elif system_name == "Brachistochrone" :
            self.s_dim = 2
            self.u_dim = 1
            self.depth = 2
            self.n_train, self.n_test = 5000, 2000
            self.s_range = 1
            self.s_offset = 0
            self.s_out_offset = 0.9
            self.s_out_range = 0.9
            self.n_epoch = 5000
            self.T = 2
            self.n_sensor = 101
            self.dt =  self.T/(self.n_sensor-1)
        
        elif system_name =="Zermelo"  :
            self.s_dim = 7
            self.u_dim = 1
            self.depth = 3
            self.n_train, self.n_test = 100000, 2000
            self.s_range = 1
            self.s_offset = 0
            self.s_out_offset = 1
            self.s_out_range = 1
            self.n_epoch = 5000
            self.T = 2
            self.n_sensor = 101
            self.dt =  self.T/(self.n_sensor-1)


        
            
        
        else:
            raise NotImplementedError("System {} is not implemented".format(system_name))

        if system_name in ["Pendulum", "RobotArm", "CartPole", "Quadrotor", "Rocket"]:
            self.input_dim = 0
            if 't' in self.mode:
                self.input_dim += self.s_dim
            if 'i' in self.mode:
                self.input_dim += self.i_dim
            if 'd' in self.mode:
                self.input_dim += self.d_dim

        else:
            self.input_dim = self.s_dim

            

        if isinstance(self.model_name, str) and "DeepControlNet" in self.model_name:
            if self.Np is None:
                self.branch_width = [self.input_dim] + [self.width]*self.depth
                self.trunk_width = [self.t_dim] + [self.width]*self.depth
            else:
                self.branch_width = [self.input_dim] + [self.width]*(self.depth-1) + [self.Np]
                self.trunk_width = [self.t_dim] + [self.width]*(self.depth-1)  + [self.Np]
        
        elif isinstance(self.model_name, str) and self.model_name == "NSMControl":
            self.branch_width = [self.input_dim] + [self.width]*self.depth
            self.trunk_width = [self.t_dim] + [self.width]*self.depth
            self.prune_epoch_ratio = 0.8  # do pruning at the ratio*n_epoch epoch.
        elif isinstance(self.model_name, str) and self.model_name == "NSMControl1":
            self.branch_width = [self.input_dim] + [self.width]*self.depth 
            self.trunk_width = [self.t_dim] + [self.width]*self.depth 
            self.prune_epoch_ratio = 0.1  # do pruning after the ratio*n_epoch epoch.
        elif isinstance(self.model_name, str) and self.model_name.startswith("NSMControl"): #NSM2, NSM2_1, NSM3, NSM4,...
            if model_name in ["NSMControl8", "NSMControl9", "NSMControl9_1"]: #convnet backbone
                if system_name == "Pushing":
                    self.width = 24
                else:
                    self.width = 14
            else:
                if system_name == "Pendulum":
                    self.depth = 3
                    self.width = 40
                elif system_name == "Pushing":
                    self.depth = 4
                    self.width = 50
                    
                elif system_name in ["Quadrotor", "Rocket"]:
                    self.n_epoch = 1000
                
                elif system_name in ["Brachistochrone", "Zermelo"]:
                    self.depth += 1

            
            if self.Np is None:
                self.Np = 11
            self.branch_width = [self.input_dim] + [self.width]*(self.depth-1) + [self.Np]
            self.trunk_width = [self.t_dim] + [self.width]*(self.depth-1)  + [self.Np]
            self.prune_epoch_ratio = 1  # no pruning
        elif self.model_name in ["MLPControl-dp", "MLPControl"]:
            # keep MLPControl num params same with DCN
            if system_name in ["Quadrotor"]:
                coef = 2.2
            elif system_name in ["Rocket"]:
                coef = 2
            elif system_name == "Pushing":
                coef = 1
            elif system_name.startswith("Pushing_"):
                coef = 1
            elif system_name in ["Pendulum", "RobotArm", "Brachistochrone", "Zermelo"]:
                coef = 1.5

            elif system_name in ["CartPole", "StochasticPendulum", "Heating"]:
                coef = 1.4
            self.mlp_width = [self.input_dim+self.t_dim] + [int(coef*self.width)]*self.depth + [self.u_dim]

    
        self.n_print_epoch = self.n_epoch // self.n_print

        



