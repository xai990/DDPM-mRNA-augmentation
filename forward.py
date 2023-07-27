from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils import gather

import umap
import matplotlib.pyplot as plt


class DenoiseDiffusion(nn.Module):

    def __init__(self,
                model: nn.Module,
                n_steps: int,
                device:torch.device):
        super().__init__()
        self.beta = torch.linspace(1e-4,0.02,n_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha,dim=0)
        self.model = model
        self.sigma2 = self.beta # sigma 2 == beta has the same performance
        self.n_steps = n_steps


    def q_xt_x0(self, x0, t):
        mean = gather(self.alpha_bar,t) ** 0.5 * x0

        var = 1 - gather(self.alpha_bar,t)
        return mean, var


    def q_sample(self, x0:torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):

        if eps == None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0,t)

        return  mean + var ** 0.5 * eps


    def p_sample(self, xt:torch.Tensor, t:torch.Tensor):

        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        beta = gather(self.beta, t)
        # eps theta
        eps_theta = self.model(xt,t)
        # coeff for eps
        eps_coef = beta / (1-alpha_bar) ** .5
        # mean
        mean =  1 / alpha ** .5 ( xt - eps_coef * eps_theta)
        # variation
        var = beta
        # random generate noise
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps

    def loss(self, x0:torch.Tensor, noise:Optional[torch.Tensor]=None):

        batch_size = x0.shape[0]
        t = torch.randint(0,self.n_steps, (batch_size,), device=x0.device,dtype=torch.long)
        # print("t size is:",t.size())
        if noise == None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, eps = noise) # (batch_size, channels, rows, cols)
        # print("xt size is:", xt.size())
        t = t.unsqueeze(-1).unsqueeze(-1) # (batch_size, nums) -> (batch_size, channels, rows, nums)
        eps_theta = self.model(xt,t)
        return F.mse_loss(noise, eps_theta)


    def plotumap(self,x0, noise:Optional[torch.Tensor]=None, num_shows:Optional[int]=20, cols:Optional[int]=10) -> None:
        # compute embedding of merge data
        reducer = umap.UMAP()

        # initialize figure
        rows = num_shows//cols
        print("The type of row is:",type(rows))
        fig,axs = plt.subplots(rows,cols,figsize=(28,3))
        plt.rc('text', color = 'black') # add text to the plot

        for i in range(num_shows):
            j = i //cols # plot col index
            k = i % cols # plot row index
            # generate q_sample
            t = torch.full((x0.shape[0],),i*self.n_steps//num_shows)
            xt = self.q_sample(x0,t)
            # apply umap embedding
            qi = reducer.fit_transform(xt[:,0,0,:]) # data type: tuple
            axs[j,k].scatter(qi[:,0],qi[:,1], color='red',edgecolor='white')
            axs[j,k].set_axis_off()
            axs[j,k].set_title('$q(\mathbf{x}_{'+str(i*self.n_steps//num_shows)+'})$')

        plt.savefig("foward.umap.png")
        plt.close()
