import sys
import time
from functools import wraps
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

class Redirect_Stdout_Null(object):
    def __init__(self):
        pass

    def __enter__(self):
        f = open('/dev/null', 'w')
        self.stdout_old = sys.stdout
        sys.stdout = f

    def __exit__(self):
        sys.stdout = self.stdout_old

def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        dt = te - ts
        if dt < 60:
            t = "{:.4f} sec.".format(dt)
        elif dt < 3600:
            t = "{:.4f} min.".format(dt/60)
        else:
            t = "{:.4f} hour.".format(dt/3600)

        print("%r took %s \n" % (f.__name__, t))
        sys.stdout.flush()
        return result

    return wrapper


def redirect_log_file(log_root = "./log"):
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    t = str(datetime.datetime.now())
    out_file = os.path.join(log_root, t)
    print("Redirect log to: ", out_file, flush=True)
    sys.stdout = open(out_file, 'a')
    sys.stderr = open(out_file, 'a')
    print("Start time:", t, flush=True)

def draw(x, y_pred, y_ref, name, fig_dir):
    fig, ax = plt.subplots()
    
    if len(y_pred.shape) == 1:
        ax.plot(x, y_pred, label="Pred")
        ax.plot(x, y_ref, label="Ref")
    elif len(y_pred.shape) == 2: # multiple ys
        for i in range(y_pred.shape[1]):
            ax.plot(x, y_pred[:,i], label="Pred_%d"%i)
            ax.plot(x, y_ref[:,i], label="Ref_%d"%i)

    if np.abs(y_pred).max()>20: # abnormal y_pred
        ax.set_yscale('log')

    plt.legend()
    plt.xlabel("t")
    plt.title("u(t)="+name)
    
    fig_path = os.path.join(fig_dir, name+".png")
    plt.savefig(fig_path, dpi =100)
    plt.close()


def draw_s(x, y_pred, y_ref, y_env, name, fig_dir):
    fig, ax = plt.subplots()
    
    if len(y_pred.shape) == 1:
        ax.plot(x, y_pred, label="Pred")
        ax.plot(x, y_ref, label="Ref")
        ax.plot(x, y_env, label="Env")
    elif len(y_pred.shape) == 2: # multiple ys
        for i in range(y_pred.shape[1]):
            ax.plot(x, y_pred[:, i], label="Pred_%d"%i)
            ax.plot(x, y_ref[:, i], label="Ref_%d"%i)
            ax.plot(x, y_env[:, i], label="Env_%d"%i)

    if np.abs(y_pred).max()>20: # abnormal y_pred
        ax.set_yscale('log')

    plt.legend()
    plt.xlabel("t")
    plt.title("s of u(t)="+name)
    
    fig_path = os.path.join(fig_dir, name+"_s.png")
    plt.savefig(fig_path, dpi =100)
    plt.close()


def draw_ys(x, ys, fig_name, fig_dir):
    """general draw function
    ys : dict of (str,array) pairs, denoting name and values
    fig_name: used as title and filename
    """
    fig, ax = plt.subplots()
    y_shape = list(ys.values())[0].shape
    line_style_dict = {"True":"solid", "DCN":"dashed", "MLP":"dashed", "TwoPhases":"dashed", "PDP":"dotted"}
    line_style_dict.setdefault("DefaultKey", "dashed")
    if len(y_shape) == 1:
        for name, y in ys.items():
            ax.plot(x, y, label=name, linestyle=line_style_dict.get(name, "dashed"))
        
    elif len(y_shape) == 2: # multiple ys
        for i in range(y_shape[1]):
            for name, y in ys.items():
                ax.plot(x, y[:, i], label="{}_{}".format(name,i), linestyle=line_style_dict.get(name, "dashed"))
            
        
    for name, y in ys.items():
        if np.abs(y).max()>200: # abnormal y_pred
            ax.set_yscale('log')

    plt.legend()
    plt.xlabel("t")
    plt.title(fig_name)
    
    fig_path = os.path.join(fig_dir, fig_name+".png")
    plt.savefig(fig_path, dpi =100)
    plt.close()

def draw_3d(X, Y, Z, fig_name, fig_dir):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig_path = os.path.join(fig_dir, fig_name+".png")
    plt.savefig(fig_path, dpi =100)
    plt.close()

