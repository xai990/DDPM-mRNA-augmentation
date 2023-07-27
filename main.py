#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from forward import DenoiseDiffusion
from MLPdiffusion import MLPDiffusion

# import umap
# import matplotlib.pyplot as plt

if __name__ == "__main__":
    # define a time step
    steps_num = 100
    # device cuda or cpu
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu").
    device = torch.device("cpu")
    # create a random x0 and t
    batch_size = 32
    dataset = torch.rand(200,1,1,20).to(device).float()
    # define the time index
    t = torch.randint(0, steps_num, size = (200//2,)).to(device)
    t = torch.cat([t, steps_num - 1 - t], dim = 0)
    # t = t.unsqueeze(-1)
    # define noise
    noise = torch.randn_like(dataset)

    # define a dataloader for traning
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoch = 1000
    print("input dim:",dataset.size()[-1])

    # print the original plot
    # reducer = umap.UMAP()
    # x0 = reducer.fit_transform(dataset[:,0,0,:])
    # fig,axs = plt.subplots(rows,cols,figsize=(28,3))
    # plt.rc('text', color = 'black') # add text to the plot
    # axs.scatter(x0[:,0],x0[:,1], color='red',edgecolor='white')
    # axs.set_axis_off()
    # axs.set_title('$q(\mathbf{x}_{'+str(0)+'})$')
    #
    # plt.savefig("dataset.umap.png")
    # plt.close()

    # DF model
    model = MLPDiffusion(dataset.size()[-1], dataset.size()[-1], steps_num)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    DF = DenoiseDiffusion(model, steps_num, device)

    # forward plot
    DF.plotumap(dataset)

    # train the DF model
    for t in range(num_epoch):
        for idx, batch_x in enumerate(dataloader): #(32,1,10,5)
            # print("batch_x size is:", batch_x.size())
            loss = DF.loss(batch_x)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

        if(t%100==0):
            print(loss)
