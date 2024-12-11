"""
# This module is partially adapted from following project
# Github:https://github.com/wanxinjin/Pontryagin-Differentiable-Programming
# Paper: Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework https://arxiv.org/abs/1912.12970

"""
import warnings
warnings.filterwarnings("ignore")
import math
import time

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
#from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch








# inverted pendulum
class SinglePendulum:
    def __init__(self, project_name='single pendlumn system'):
        self.project_name = project_name

    def initDyn(self, l=None, m=None, damping_ratio=None):
        g=10
        # declare system parameter
        parameter = []
        if l is None:
            self.l = SX.sym('l')
            parameter += [self.l]
        else:
            self.l = l

        if m is None:
            self.m = SX.sym('m')
            parameter += [self.m]
        else:
            self.m = m

        if damping_ratio is None:
            self.damping_ratio = SX.sym('damping_ratio')
            parameter += [self.damping_ratio]
        else:
            self.damping_ratio = damping_ratio

        self.dyn_auxvar = vcat(parameter)

        # set variable
        self.q, self.dq = SX.sym('q'), SX.sym('dq')
        self.X = vertcat(self.q, self.dq)
        U = SX.sym('u')
        self.U = U
        I = 1 / 3 * self.m * self.l * self.l  
        self.f = vertcat(self.dq,
                         (self.U - self.m * g * self.l * sin(
                             self.q) - self.damping_ratio * self.dq) / I)  # continuous state-space representation

    def initCost(self, wq=None, wdq=None, wu=0.001, x_goal=[math.pi, 0]):
        parameter = []
        if wq is None:
            self.wq = SX.sym('wq')
            parameter += [self.wq]
        else:
            self.wq = wq

        if wdq is None:
            self.wdq = SX.sym('wdq')
            parameter += [self.wdq]
        else:
            self.wdq = wdq

        self.cost_auxvar = vcat(parameter)
        self.wu = wu
        self.setTargetState(x_goal)

       
    def setTargetState(self, x_goal):
        # cost for q
        self.cost_q = (self.q - x_goal[0]) ** 2
        # cost for dq
        self.cost_dq = (self.dq - x_goal[1]) ** 2
        # cost for u
        self.cost_u = dot(self.U, self.U)

        self.path_cost = self.wq * self.cost_q + self.wdq * self.cost_dq + self.wu * self.cost_u



# robot arm environment
class RobotArm:

    def __init__(self, project_name='two-link robot arm'):
        self.project_name = project_name

    def initDyn(self, l1=None, m1=None, l2=None, m2=None ):
        g=0
        # declare system parameters
        parameter = []
        if l1 is None:
            self.l1 = SX.sym('l1')
            parameter += [self.l1]
        else:
            self.l1 = l1

        if m1 is None:
            self.m1 = SX.sym('m1')
            parameter += [self.m1]
        else:
            self.m1 = m1

        if l2 is None:
            self.l2 = SX.sym('l2')
            parameter += [self.l2]
        else:
            self.l2 = l2

        if m2 is None:
            self.m2 = SX.sym('m2')
            parameter += [self.m2]
        else:
            self.m2 = m2

        self.dyn_auxvar = vcat(parameter)

        # set variable
        self.q1, self.dq1, self.q2, self.dq2 = SX.sym('q1'), SX.sym('dq1'), SX.sym('q2'), SX.sym('dq2')
        self.X = vertcat(self.q1, self.q2, self.dq1, self.dq2)
        #u1, u2 = SX.sym('u1'), SX.sym('u2') #set u1 to zero.
        #self.U = vertcat(u1, u2)
        self.U = SX.sym('u2')

        # Declare model equations (discrete-time)
        r1 = self.l1 / 2
        r2 = self.l2 / 2
        I1 = self.l1 * self.l1 * self.m1 / 3
        I2 = self.l2 * self.l2 * self.m2 / 3
        M11 = self.m1 * r1 * r1 + I1 + self.m2 * (self.l1 * self.l1 + r2 * r2 + 2 * self.l1 * r2 * cos(self.q2)) + I2
        M12 = self.m2 * (r2 * r2 + self.l1 * r2 * cos(self.q2)) + I2
        M21 = M12
        M22 = self.m2 * r2 * r2 + I2
        M = vertcat(horzcat(M11, M12), horzcat(M21, M22))
        h = self.m2 * self.l1 * r2 * sin(self.q2)
        C1 = -h * self.dq2 * self.dq2 - 2 * h * self.dq1 * self.dq2
        C2 = h * self.dq1 * self.dq1
        C = vertcat(C1, C2)
        G1 = self.m1 * r1 * g * cos(self.q1) + self.m2 * g * (r2 * cos(self.q1 + self.q2) + self.l1 * cos(self.q1))
        G2 = self.m2 * g * r2 * cos(self.q1 + self.q2)
        G = vertcat(G1, G2)
        ddq = mtimes(inv(M), -C - G + SX([0,1])*self.U)  # joint acceleration, only at elbow
        self.f = vertcat(self.dq1, self.dq2, ddq)  # continuous state-space representation

    def initCost(self, wq1=None, wq2=None, wdq1=None, wdq2=None, wu=0.1, x_goal = [math.pi / 2, 0, 0, 0]):
        # declare system parameters
        parameter = []
        if wq1 is None:
            self.wq1 = SX.sym('wq1')
            parameter += [self.wq1]
        else:
            self.wq1 = wq1

        if wq2 is None:
            self.wq2 = SX.sym('wq2')
            parameter += [self.wq2]
        else:
            self.wq2 = wq2

        if wdq1 is None:
            self.wdq1 = SX.sym('wdq1')
            parameter += [self.wdq1]
        else:
            self.wdq1 = wdq1

        if wdq2 is None:
            self.wdq2 = SX.sym('wdq2')
            parameter += [self.wdq2]
        else:
            self.wdq2 = wdq2

        self.cost_auxvar = vcat(parameter)

        self.wu = wu
        self.setTargetState(x_goal)

       
    def setTargetState(self, x_goal):
        # cost for q1
        self.cost_q1 = (self.q1 - x_goal[0]) ** 2
        # cost for q2
        self.cost_q2 = (self.q2 - x_goal[1]) ** 2
        # cost for dq1
        self.cost_dq1 = (self.dq1 - x_goal[2]) ** 2
        # cost for q2
        self.cost_dq2 = (self.dq2 - x_goal[3]) ** 2
        # cost for u
        self.cost_u = dot(self.U, self.U)

        self.path_cost = self.wq1 * self.cost_q1 + self.wq2 * self.cost_q2 + \
                         self.wdq1 * self.cost_dq1 + self.wdq2 * self.cost_dq2 + self.wu * self.cost_u



# Cart Pole environment
class CartPole:
    def __init__(self, project_name='cart-pole-system'):
        self.project_name = project_name

    def initDyn(self, mc=None, mp=None, l=None):
        # set the global parameters
        g = 10

        # declare system parameters
        parameter = []
        if mc is None:
            self.mc = SX.sym('mc')
            parameter += [self.mc]
        else:
            self.mc = mc

        if mp is None:
            self.mp = SX.sym('mp')
            parameter += [self.mp]
        else:
            self.mp = mp
        if l is None:
            self.l = SX.sym('l')
            parameter += [self.l]
        else:
            self.l = l
        self.dyn_auxvar = vcat(parameter)

        # Declare system variables
        self.x, self.q, self.dx, self.dq = SX.sym('x'), SX.sym('q'), SX.sym('dx'), SX.sym('dq')
        self.X = vertcat(self.x, self.q, self.dx, self.dq)
        self.U = SX.sym('u')
        ddx = (self.U + self.mp * sin(self.q) * (self.l * self.dq * self.dq + g * cos(self.q))) / (
                self.mc + self.mp * sin(self.q) * sin(self.q))  # acceleration of x
        ddq = (-self.U * cos(self.q) - self.mp * self.l * self.dq * self.dq * sin(self.q) * cos(self.q) - (
                self.mc + self.mp) * g * sin(
            self.q)) / (
                      self.l * self.mc + self.l * self.mp * sin(self.q) * sin(self.q))  # acceleration of theta
        self.f = vertcat(self.dx, self.dq, ddx, ddq)  # continuous dynamics

    def initCost(self, wx=None, wq=None, wdx=None, wdq=None, wu=0.001, x_goal = [0.0, math.pi, 0.0, 0.0]):
        # declare system parameters
        parameter = []
        if wx is None:
            self.wx = SX.sym('wx')
            parameter += [self.wx]
        else:
            self.wx = wx

        if wq is None:
            self.wq = SX.sym('wq')
            parameter += [self.wq]
        else:
            self.wq = wq
        if wdx is None:
            self.wdx = SX.sym('wdx')
            parameter += [self.wdx]
        else:
            self.wdx = wdx

        if wdq is None:
            self.wdq = SX.sym('wdq')
            parameter += [self.wdq]
        else:
            self.wdq = wdq
        self.cost_auxvar = vcat(parameter)

        self.wu = wu
        self.setTargetState(x_goal)

       
    def setTargetState(self, x_goal):
        

        self.path_cost = self.wx * (self.x - x_goal[0]) ** 2 + self.wq * (self.q - x_goal[1]) ** 2 + self.wdx * (
                self.dx - x_goal[2]) ** 2 + self.wdq * (
                                 self.dq - x_goal[3]) ** 2 + self.wu * (self.U * self.U)




# quadrotor (UAV) environment
class Quadrotor:
    def __init__(self, project_name='my UAV'):
        self.project_name = 'my uav'
        # See PDP paper appendix E.1 for definition
        # state: r(position), v(velocity), q(attitude), w(angular velocity)
        # control: T(thrusts(forces) of 4 rotating propeller)

        # define the state of the quadrotor
        rx, ry, rz = SX.sym('rx'), SX.sym('ry'), SX.sym('rz')
        self.r_I = vertcat(rx, ry, rz)
        vx, vy, vz = SX.sym('vx'), SX.sym('vy'), SX.sym('vz')
        self.v_I = vertcat(vx, vy, vz)
        # quaternions attitude of B w.r.t. I
        q0, q1, q2, q3 = SX.sym('q0'), SX.sym('q1'), SX.sym('q2'), SX.sym('q3')
        self.q = vertcat(q0, q1, q2, q3)
        wx, wy, wz = SX.sym('wx'), SX.sym('wy'), SX.sym('wz')
        self.w_B = vertcat(wx, wy, wz)
        # define the quadrotor input
        f1, f2, f3, f4 = SX.sym('f1'), SX.sym('f2'), SX.sym('f3'), SX.sym('f4')
        self.T_B = vertcat(f1, f2, f3, f4)

    def initDyn(self, Jx=None, Jy=None, Jz=None, mass=None, l=None):
        # global parameter
        g = 10
        c = 0.01

        # parameters settings
        parameter = []
        if Jx is None:
            self.Jx = SX.sym('Jx')
            parameter += [self.Jx]
        else:
            self.Jx = Jx

        if Jy is None:
            self.Jy = SX.sym('Jy')
            parameter += [self.Jy]
        else:
            self.Jy = Jy

        if Jz is None:
            self.Jz = SX.sym('Jz')
            parameter += [self.Jz]
        else:
            self.Jz = Jz

        if mass is None:
            self.mass = SX.sym('mass')
            parameter += [self.mass]
        else:
            self.mass = mass

        if l is None:
            self.l = SX.sym('l')
            parameter += [self.l]
        else:
            self.l = l

        if c is None:
            self.c = SX.sym('c')
            parameter += [self.c]
        else:
            self.c = c

        self.dyn_auxvar = vcat(parameter)

        # Angular moment of inertia
        self.J_B = diag(vertcat(self.Jx, self.Jy, self.Jz))
        # Gravity
        self.g_I = vertcat(0, 0, -g)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        # total thrust in body frame
        thrust = self.T_B[0] + self.T_B[1] + self.T_B[2] + self.T_B[3]
        self.thrust_B = vertcat(0, 0, thrust)
        # total moment M in body frame
        Mx = -self.T_B[1] * self.l / 2 + self.T_B[3] * self.l / 2
        My = -self.T_B[0] * self.l / 2 + self.T_B[2] * self.l / 2
        Mz = (self.T_B[0] - self.T_B[1] + self.T_B[2] - self.T_B[3]) * self.c
        self.M_B = vertcat(Mx, My, Mz)

        # cosine directional matrix
        C_B_I = self.dir_cosine(self.q)  # inertial to body
        C_I_B = transpose(C_B_I)  # body to inertial

        # Newton's law
        dr_I = self.v_I
        dv_I = 1 / self.m * mtimes(C_I_B, self.thrust_B) + self.g_I
        # Euler's law
        dq = 1 / 2 * mtimes(self.omega(self.w_B), self.q)
        dw = mtimes(inv(self.J_B), self.M_B - mtimes(mtimes(self.skew(self.w_B), self.J_B), self.w_B))

        self.X = vertcat(self.r_I, self.v_I, self.q, self.w_B)
        self.U = self.T_B
        self.f = vertcat(dr_I, dv_I, dq, dw)

    def initCost(self, wr=None, wv=None, wq=None, ww=None, wthrust=0.1, x_goal=np.zeros((9,))):

        parameter = []
        if wr is None:
            self.wr = SX.sym('wr')
            parameter += [self.wr]
        else:
            self.wr = wr

        if wv is None:
            self.wv = SX.sym('wv')
            parameter += [self.wv]
        else:
            self.wv = wv

        if wq is None:
            self.wq = SX.sym('wq')
            parameter += [self.wq]
        else:
            self.wq = wq

        if ww is None:
            self.ww = SX.sym('ww')
            parameter += [self.ww]
        else:
            self.ww = ww
        self.wthrust = wthrust  # thrust = u
        self.cost_auxvar = vcat(parameter)

        self.setTargetState(x_goal)

       
    def setTargetState(self, x_goal):

        # goal position in the world frame
        #goal_r_I = np.array([0, 0, 0])
        goal_r_I = x_goal[0:3]
        self.cost_r_I = dot(self.r_I - goal_r_I, self.r_I - goal_r_I)

        # goal velocity
        #goal_v_I = np.array([0, 0, 0])
        goal_v_I = x_goal[3:6]
        self.cost_v_I = dot(self.v_I - goal_v_I, self.v_I - goal_v_I)

        # final attitude error
        #goal_q = toQuaternion(0, [0, 0, 1])
        #goal_R_B_I = self.dir_cosine(goal_q)
        #R_B_I = self.dir_cosine(self.q)
        #self.cost_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

        # auglar velocity cost
        #goal_w_B = np.array([0, 0, 0])
        goal_w_B = x_goal[6:9]
        self.cost_w_B = dot(self.w_B - goal_w_B, self.w_B - goal_w_B)

        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)


        self.path_cost = self.wr * self.cost_r_I + \
                         self.wv * self.cost_v_I + \
                         self.ww * self.cost_w_B + \
                         self.wthrust * self.cost_thrust \
                         #self.wq * self.cost_q + 
    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg

    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                       )



# Rocket environment
class Rocket:
    def __init__(self, project_name='rocket powered landing'):
        self.project_name = project_name

        # define the rocket states
        rx, ry, rz = SX.sym('rx'), SX.sym('ry'), SX.sym('rz')
        self.r_I = vertcat(rx, ry, rz)
        vx, vy, vz = SX.sym('vx'), SX.sym('vy'), SX.sym('vz')
        self.v_I = vertcat(vx, vy, vz)
        # quaternions attitude of B w.r.t. I
        q0, q1, q2, q3 = SX.sym('q0'), SX.sym('q1'), SX.sym('q2'), SX.sym('q3')
        self.q = vertcat(q0, q1, q2, q3)
        wx, wy, wz = SX.sym('wx'), SX.sym('wy'), SX.sym('wz')
        self.w_B = vertcat(wx, wy, wz)
        # define the rocket input
        ux, uy, uz = SX.sym('ux'), SX.sym('uy'), SX.sym('uz')
        self.T_B = vertcat(ux, uy, uz)

    def initDyn(self, Jx=None, Jy=None, Jz=None, mass=None, l=None):
        # global parameter
        g = 10

        # parameters settings
        parameter = []
        if Jx is None:
            self.Jx = SX.sym('Jx')
            parameter += [self.Jx]
        else:
            self.Jx = Jx

        if Jy is None:
            self.Jy = SX.sym('Jy')
            parameter += [self.Jy]
        else:
            self.Jy = Jy

        if Jz is None:
            self.Jz = SX.sym('Jz')
            parameter += [self.Jz]
        else:
            self.Jz = Jz

        if mass is None:
            self.mass = SX.sym('mass')
            parameter += [self.mass]
        else:
            self.mass = mass

        if l is None:
            self.l = SX.sym('l')
            parameter += [self.l]
        else:
            self.l = l

        self.dyn_auxvar = vcat(parameter)

        # Angular moment of inertia
        self.J_B = diag(vertcat(self.Jx, self.Jy, self.Jz))
        # Gravity
        self.g_I = vertcat(-g, 0, 0)
        # Vector from thrust point to CoM
        self.r_T_B = vertcat(-self.l / 2, 0, 0)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        C_B_I = self.dir_cosine(self.q)
        C_I_B = transpose(C_B_I)

        dr_I = self.v_I
        dv_I = 1 / self.m * mtimes(C_I_B, self.T_B) + self.g_I

        dq = 1 / 2 * mtimes(self.omega(self.w_B), self.q)
        dw = mtimes(inv(self.J_B),
                    mtimes(self.skew(self.r_T_B), self.T_B) -
                    mtimes(mtimes(self.skew(self.w_B), self.J_B), self.w_B))

        self.X = vertcat(self.r_I, self.v_I, self.q, self.w_B)
        self.U = self.T_B
        self.f = vertcat(dr_I, dv_I, dq, dw)

    def initCost(self, wr=None, wv=None, wtilt=None, ww=None, wsidethrust=None, wthrust=1.0, x_goal=np.zeros((9,))):

        parameter = []
        if wr is None:
            self.wr = SX.sym('wr')
            parameter += [self.wr]
        else:
            self.wr = wr

        if wv is None:
            self.wv = SX.sym('wv')
            parameter += [self.wv]
        else:
            self.wv = wv

        if wtilt is None:
            self.wtilt = SX.sym('wtilt')
            parameter += [self.wtilt]
        else:
            self.wtilt = wtilt

        if wsidethrust is None:
            self.wsidethrust = SX.sym('wsidethrust')
            parameter += [self.wsidethrust]
        else:
            self.wsidethrust = wsidethrust

        if ww is None:
            self.ww = SX.sym('ww')
            parameter += [self.ww]
        else:
            self.ww = ww

        self.wthrust = wthrust

        self.cost_auxvar = vcat(parameter)

        self.setTargetState(x_goal)

       
    def setTargetState(self, x_goal):

        # goal position in the world frame
        #goal_r_I = np.array([0, 0, 0])
        goal_r_I = x_goal[0:3]
        self.cost_r_I = dot(self.r_I - goal_r_I, self.r_I - goal_r_I)

        # goal velocity
        #goal_v_I = np.array([0, 0, 0])
        goal_v_I = x_goal[3:6]
        self.cost_v_I = dot(self.v_I - goal_v_I, self.v_I - goal_v_I)

        # tilt angle upward direction of rocket should be close to upward of earth
        #C_I_B = transpose(self.dir_cosine(self.q))
        #nx = np.array([1., 0., 0.]) 
        #ny = np.array([0., 1., 0.]) 
        #nz = np.array([0., 0., 1.]) 
        #proj_ny = dot(ny, mtimes(C_I_B, nx))
        #proj_nz = dot(nz, mtimes(C_I_B, nx))
        #self.cost_tilt = proj_ny ** 2 + proj_nz ** 2

        # the sides of the thrust should be zeros
        #self.cost_side_thrust = (self.T_B[1] ** 2 + self.T_B[2] ** 2)

        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        # auglar velocity cost
        #goal_w_B = np.array([0, 0, 0])
        goal_w_B = x_goal[6:9]
        self.cost_w_B = dot(self.w_B - goal_w_B, self.w_B - goal_w_B)

        self.path_cost = self.wr * self.cost_r_I + \
                         self.wv * self.cost_v_I + \
                         self.ww * self.cost_w_B + \
                         self.wthrust * self.cost_thrust
                         #self.wtilt * self.cost_tilt + 
                         #self.wsidethrust * self.cost_side_thrust + 
    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg



# converter to quaternion from (angle, direction)
def toQuaternion(angle, dir):
    if type(dir) == list:
        dir = numpy.array(dir)
    dir = dir / numpy.linalg.norm(dir)
    quat = numpy.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat.tolist()


# normalized verctor
def normalizeVec(vec):
    if type(vec) == list:
        vec = np.array(vec)
    vec = vec / np.linalg.norm(vec)
    return vec


def quaternion_conj(q):
    conj_q = q
    conj_q[1] = -q[1]
    conj_q[2] = -q[2]
    conj_q[3] = -q[3]
    return conj_q

'''
class StochasticPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g, l, m, damping_ratio, dt, wq, wdq, wu, x_goal, x_init, seed=None):

        
        # time interval and horizon
        self.dt = dt
        self.T = 1

        #dynamics params
        self.g = g
        self.l = l
        self.m = m
        self.damping_ratio = damping_ratio
        self.x_init = x_init

        # cost params
        self.wq=wq 
        self.wdq = wdq 
        self.wu = wu 
        self.x_goal = x_goal

        # limits
        self.max_speed=8
        self.max_torque=2.
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        low = np.array([ -1., -1., -self.max_speed], dtype=np.float32)# obs=[sin(q), cos(q), dq]
        high = np.array([ 1., 1., self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32) 

        # randomness, noise
        self.seed(seed)
        #self.noise_scale = 1.0
        self.noise_scale = 0.01

        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def angle_normalize(self,x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def step(self, action):
        q, dq = self.state 
        action = np.clip(action, -1, 1)[0]
        u = action * self.max_torque  # rescale of u.  'action' is for RL model, 'u' is the actual control

        I = 1 / 3 * self.m * self.l * self.l  
        
        f0 = dq
        f1 = (u - self.m * self.g * self.l * np.sin(q) - self.damping_ratio * dq) / I

        #next state
        q_next = q + self.dt * f0
        noise = self.np_random.normal(loc=0.0, scale=self.noise_scale**2 * self.dt)
        dq_next = dq + self.dt *f1 + noise
        dq_next = np.clip(dq_next, -self.max_speed, self.max_speed)
        self.state = np.array([ q_next, dq_next])
        
        # costs
        costs = self.wq * (q -self.x_goal[0])**2 + self.wdq * (dq -self.x_goal[1])**2 + self.wu * u**2

        # returning 
        observation = self._get_obs()
        reward = -costs
        done = False
        info = {}
        return observation, reward, done, info

    def reset(self):
        self.state = np.array(self.x_init)
        return self._get_obs()

    def _get_obs(self):
        q, dq = self.state
        return np.array([np.cos(q), np.sin(q), dq], dtype=np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
'''

def get_env(name):
    if name in ["Pendulum", "RobotArm", "CartPole", "Quadrotor", "Rocket"]:
        if name == "Pendulum":
            env = SinglePendulum()
            #dyn_args = {"l":1, "m":1, "damping_ratio":0.05}
            dyn_args = [1, 1, 0.05]
            env.initDyn(*dyn_args)
            wq=10
            wdq=1
            wu=0.1
            x_goal = [math.pi, 0]
            env.initCost(wq=wq, wdq=wdq, wu=wu, x_goal=x_goal)
            ini_state = [0, 0]
            loss_weight = [wq, wdq, wu]
        elif name == "RobotArm":
            env = RobotArm()
            #dyn_args = {"l1":1, "m1":1, "l2":1, "m2":1}
            dyn_args = [1,1,1,1]
            env.initDyn(*dyn_args)
            wq1, wq2, wdq1, wdq2, wu = 0.1, 0.1, 0.1, 0.1, 0.01
            loss_weight = [wq1, wq2, wdq1, wdq2, wu]
            x_goal = [math.pi / 2, 0, 0, 0]
            env.initCost(wq1=wq1, wq2=wq2, wdq1=wdq1, wdq2=wdq2, wu=wu, x_goal=x_goal)
            ini_state = [pi/4, pi/2, 0, 0]
            
        elif name == "CartPole":
            env = CartPole()
            #dyn_args = {"mc":0.1, "mp":0.1, "l":1}
            dyn_args = [0.1, 0.1, 1]
            env.initDyn(*dyn_args)
            wx, wq, wdx, wdq, wu = 0.1, 0.6, 0.1, 0.1, 0.3
            x_goal = [0.0, math.pi, 0.0, 0.0]
            env.initCost(wx=wx, wq=wq, wdx=wdx, wdq=wdq, wu=wu, x_goal=x_goal)
            ini_state = [0, 0, 0, 0]
            loss_weight = [wx, wq, wdx, wdq, wu]

        elif name == "Quadrotor":
            env = Quadrotor()
            #dyn_args = {"Jx":1, "Jy":1, "Jz":1, "mass":1, "win_len":0.4} 
            dyn_args = [1,1,1,1,0.4]
            env.initDyn(*dyn_args)
            wr, wv, ww, wu = 1, 1, 1, 0.1
            loss_weight = [wr, wv, ww, wu] 
            #NOTE: loss weights here are used for vectors of state, not scalars of state.

            x_goal = np.zeros((9,))
            env.initCost(wr=wr, wv=wv, wq=0, ww=ww, wthrust=wu, x_goal=x_goal)

            ini_r_I = [-8, -6, 9.]
            ini_v_I = [0.0, 0.0, 0.0]
            ini_q = toQuaternion(0, [1, -1, 1])
            ini_w = [0.0, 0.0, 0.0]
            ini_state = ini_r_I + ini_v_I + ini_q + ini_w
        
        elif name == "Rocket":
            env = Rocket()
            #dyn_args = {"Jx":0.5, "Jy":1., "Jz":1., "mass":1., "l":1.}
            dyn_args = [0.5, 1, 1, 1, 1]
            env.initDyn(*dyn_args)
            x_goal = np.zeros((9,))
            env.initCost(wr=1, wv=1, wtilt=0, ww=1, wsidethrust=0, wthrust=0.4, x_goal=x_goal)
            loss_weight = [1, 1, 1, 0.4]
            ini_r_I = [10, -8, 5.]
            ini_v_I = [-.1, 0.0, -0.0]
            ini_q = toQuaternion(1.5, [0, 0, 1])
            ini_w = [0, -0.0, 0.0]
            ini_state = ini_r_I + ini_v_I + ini_q + ini_w
        
        return env, ini_state, loss_weight, x_goal, dyn_args
    
  
    elif name == "Brachistochrone" or name=="Zermelo":
        # dummy 
        return None, [2,1], [1], None, None
    
    
    else:
        raise NotImplementedError("Env {} is not implemented".format(name))

    
    