import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.gaussian_process import GaussianProcessRegressor
from itertools import product

def get_sample(x, a=0.5, b=-0.2) -> np.ndarray:
    ret = a*np.cos(x) + b*np.sin(x)
    return ret

class BayesOptimizer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.orig_params = params
        self.x = []
        self.y = []
        self.ims = []
        self.fig = plt.figure()
        self.legend_flag = True
    
    def set_init_values(self, params, values):
        for p, v in zip(params, values):
            self.x.append(p)
            self.y.append(v)
            self.params.remove(p[0])
    
    def supply_next_params(self):        
        next_idx = self.get_gp_sample(np.array(self.params).reshape(-1, 1))
        next_x = self.params[next_idx]
        self.x.append([next_x])
        self.y.append(get_sample(next_x))
        self.params.remove(next_x)
    
    def get_gp_sample(self, x):
        gp = self.fit_gp()
        mean, sig = gp.predict(x, return_std=True)
        self.plot_acq(mean, sig)
        next_idx = self.acq_ucb(mean, sig)
        return next_idx
    
    def fit_gp(self):
        gp = GaussianProcessRegressor()
        gp.fit(self.x, self.y)
        return gp
    
    def acq_ucb(self, mean, sig, beta=3):
        return np.argmax(mean + sig * np.sqrt(beta))

    def plot_acq(self, mean, sig):
        upper = [m+2*s for m, s in zip(mean, sig)]
        lower = [m-2*s for m, s in zip(mean, sig)]
        im1 = plt.fill_between(self.params, upper, lower, color='orange', label='confidence95%')
        #im1 = plt.scatter(self.params, upper, color='orange', label='upper')
        #im2 = plt.scatter(self.params, lower, color='orange', label='lower')
        im3 = plt.scatter(self.orig_params, get_sample(self.orig_params), color='blue', label='true')
        im4 = plt.scatter(self.params, mean, color='green', label='mean')
        if self.legend_flag:
            plt.legend()
            self.legend_flag = False
        self.ims.append([im1, im3, im4])

    def plot_animation(self):
        ani = animation.ArtistAnimation(self.fig, self.ims, interval=500, blit=False)
        plt.show()
        
if __name__ == '__main__':
    # optimizer
    params = [-3+0.1*x for x in range(70)]
    bo = BayesOptimizer(params)

    # 初期点として2点与える
    bo.set_init_values(params=[[1], [-2]], values=[get_sample(1), get_sample(-2)])

    for _ in range(10):
        bo.supply_next_params()
    bo.plot_animation()