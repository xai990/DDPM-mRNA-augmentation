"""
Train a diffusion model on images.
"""

import argparse
from torch.utils.data import DataLoader
import torch as th

from improved_diffusion import dist_util, logger
from improved_diffusion.mrna_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    create_gaussian_diffusion,
    #dataprocess_defaults,
)

from improved_diffusion.train_util import TrainLoop
from improved_diffusion.unet import UNetModel
from improved_diffusion.gaussian_diffusion import DenoiseDiffusion

import matplotlib.pyplot as plt
import torch.onnx
import time
import numpy as np 

from torch.nn.parallel.distributed import DistributedDataParallel as DDP


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
    )
    defaults.update(model_and_diffusion_defaults())
    #defaults.update(dataprocess_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



def main():  
    start = time.time() 
    logger.configure(dir = 'log/')
    logger.log("**********************************")
    logger.log("log configure")

    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.debug(args)
    logger.debug(args_to_dict(args, model_and_diffusion_defaults().keys()))
    logger.info(f"device information:{dist_util.dev()}")

    # dataset 
    # dataset = CustomGeneDataset(**args_to_dict(args, dataprocess_defaults().keys()))
    # dataset = dataset.to_numpy()
    # dataset = dataset.reshape(dataset.shape[0],1,-1)
    logger.debug(f"batch information is: {args.batch_size} --mrna_dataset")    # data = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
    )
    # create the model 
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()) # **kwargs is to accept an arbitrary number of keyword arguments
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    # train the model 
    logger.log("traing the model ...")
    TrainLoop(
        model = model,
        diffusion = diffusion,
        data = data,
        batch_size = args.batch_size,
        microbatch = args.microbatch,
        lr = args.lr,
        ema_rate = args.ema_rate,
        log_interval = args.log_interval,
        save_interval = args.save_interval,
        resume_checkpoint= args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

    


    """
    # define a time step
    diffusion_steps = 1000
    # device cuda or cpu
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    #device = th.device("cpu")
    batch_size = 32
    # batch_size = 32 cuda out of memory 
    '''
    # create a random x0 and t
    batch_size = 32
    dataset = torch.rand(200,1,1,20).to(device).float()

    # define the time index
    t = torch.randint(0, diffusion_steps, size = (200//2,)).to(device)
    t = torch.cat([t, diffusion_steps - 1 - t], dim = 0)
    # t = t.unsqueeze(-1)
    # define noise
    noise = torch.randn_like(dataset)

    # define a dataloader for traning
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoch = 1000
    print("input features:",dataset.size()[-1])

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
    '''
    
    # s curve data 
    #dataset = s_curve_configure().to(device)
    
    #logger.debug(f"dataset:{dataset.size()}")
    # DF model
    # define the hyper parameters for Unet model
    
    in_channels = 1
    model_channels = 32
    out_channels = 1
    num_res_blocks = 1
    attention_resolutions = []
    #model = MLPDiffusion(dataset.size()[-1], dataset.size()[-1], diffusion_steps)
    # set up cuda device
    if torch.cuda.device_count() > 1:
        logger.log(f"Using {torch.cuda.device_count()} GPUs!")
        model = DDP(model)
    model = UNetModel(in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions,dims=1,).to(device)
    optimizer = th.optim.Adam(model.parameters(),lr=1e-3)
    DF = DenoiseDiffusion(model, diffusion_steps, device)
    # DF = create_gaussian_diffusion(
    #     steps=diffusion_steps,
    #     noise_schedule="linear",
    #     use_kl=False,
    # )
    # forward plot
    #DF.plotumap(dataset)
    num_shows = 20
    fig,axs = plt.subplots(2,10,figsize=(28,3))
    plt.rc('text', color='black')

    for i in range(num_shows):
        j = i // 10
        k = i % 10
        q_i = DF.q_sample(dataset, th.tensor([i*diffusion_steps//num_shows]).to(device)).cpu()
        '''if i == 0:
            mean, var = DF.q_xt_x0(dataset,torch.tensor([i*diffusion_steps//num_shows]))
            logger.log(f"The mean is:{mean}")
            logger.log(f"The var is:{var}")
            logger.log(f"q_0 is:{q_i[0]}")
            logger.log(f"dataset is:{dataset[0]}")
            assert (q_i == dataset).all(), "q0 is not the same"
        '''
        axs[j,k].scatter(q_i[:,0,0], q_i[:,0,1], color = 'red', edgecolor='white')
        axs[j,k].set_axis_off()
        axs[j,k].set_title('$q(\mathbf{x}_{'+str(i*diffusion_steps//num_shows)+'})$')
    fig.savefig('results/forward.png')
    plt.close()


    # define a dataloader for traning
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoch = 1
    
    # train the DF model
    logger.info("Training model...")
    plt.rc('text', color='black')
    for t in range(num_epoch):
        for idx, batch_x in enumerate(dataloader): #(32,1,10,5)
            # print("batch_x size is:", batch_x.size())
            #loss = DF.loss(batch_x)
            t = th.randint(0,diffusion_steps, (batch_size,), device=device)
            weights = th.tensor(np.ones([batch_size])).to(device)
            losses = DF.training_losses(model,batch_x,t)
            loss = (losses["loss"] * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

        if(t%100==0):
            #print(loss)
            logger.log(f"The training loss is:{loss} at {t} epoch")
        #     x_seq = DF.p_sample_loop(model, dataset.shape)
        #     fig,axs = plt.subplots(1,10,figsize=(28,3))
        #     for i in range(1,11):
        #         cur_x = x_seq[i*10].detach().cpu()
        #         logger.debug(f"The type of cur_x is:{type(cur_x)} and the size of cur_x is:{cur_x.shape}")
        #         axs[i-1].scatter(cur_x[:,:,0],cur_x[:,:,1],color='red',edgecolor='white');
        #         axs[i-1].set_axis_off();
        #         axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
        # fig.savefig('results/reverse.png')
        # plt.close()
    t = th.randint(0,diffusion_steps, (dataset.shape[0],), device=device,dtype=th.long)
    #th.onnx.export(model, (dataset,t), 'model.onnx', input_names=['data','t'], output_names=["noise"])
    end = time.time()
    runtime = round(end-start)
    logger.log(f"Run time:{runtime} seconds")
    """

    
if __name__ == "__main__":
    main()


# cosine beta
# mpiexec 