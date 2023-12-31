import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel

from sklearn.datasets import make_s_curve 
import matplotlib.pyplot as plt
import torch as th
from . import logger 
import os 
import umap.plot
import numpy as np 
from scipy.stats import gaussian_kde

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        feature_size=64, # origin is image_size
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        dims=2,
    )


def create_model_and_diffusion(
    feature_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    dims,
):
    model = create_model(
        feature_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        dims = dims,
        
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    feature_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    dims,
):
    # if feature_size == 256:
    #     channel_mult = (1, 1, 2, 2, 4, 4)
    # elif feature_size == 64:
    #     channel_mult = (1, 2, 3, 4)
    # elif feature_size == 32:
    #     channel_mult = (1, 2, 2, 2)
    # else:
    #     raise ValueError(f"unsupported image size: {feature_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(feature_size // int(res))

    return UNetModel(
        in_channels=1,
        model_channels=num_channels,
        out_channels=1,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        #channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dims = dims,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=1,
        model_channels=num_channels,
        out_channels=1,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")



# def s_curve_configure():
#     # s curve data 
#     s_curve,_ = make_s_curve(10**4,noise=0.1)
#     s_curve = s_curve[:,[0,2]]/10.0
#     data = s_curve.T 
#     fig, ax = plt.subplots()
#     ax.scatter(*data, color = 'blue', edgecolor='white')
#     ax.axis('off')
#     fig.savefig('results/input.png')
#     plt.close()
#     dataset = torch.Tensor(s_curve).float()
#     sample= dataset.size()[0]
#     logger.log(f"sampel is:{sample}")
#     dataset= dataset.view(sample, 1, -1)
#     #logger.log(f"dataset size is:{dataset[0]}")
#     return dataset

# def dataprocess_defaults():
#     """
#     Defaults for image training.
#     """
#     return dict(
#         genepath="datasets/breast_train_GEM_transpose.txt",
#         labelpath="datasets/train_labels.txt",
#         transform = GeneDataTransform(),
#         scaler = True,   
#     )



def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()



def plotforwardumap(diffusion,x0, n_steps = 1000, noise=None, num_shows=20, cols=10):
        # compute embedding of merge data
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        dataset = th.from_numpy(x0)
        logger.log(f"The size of dataset is {dataset.size()} --script_util")
        # initialize figure
        rows = num_shows//cols

        #print("The type of row is:",type(rows))
        fig,axs = plt.subplots(rows,cols,figsize=(28,3))
        plt.rc('text', color = 'black') # add text to the plot
        
        for i in range(num_shows):
            
            j = i //cols # plot col index
            k = i % cols # plot row index
            # generate q_sample
            t = th.full((dataset.shape[0],),i*n_steps//num_shows)
            xt = diffusion.q_sample(dataset,t)
            #logger.log(f"The type of dataset is {type(xt)} --script_util")
            # apply umap embedding
            qi = reducer.fit_transform(xt) # data type: tuple
            axs[j,k].scatter(qi[:,0],qi[:,1], color='red',edgecolor='white')
            axs[j,k].set_axis_off()
            axs[j,k].set_title('$q(\mathbf{x}_{'+str(i*n_steps//num_shows)+'})$')

        plt.savefig("results/forward.umap.png")
        plt.close()




def plotreverseumap(x0, num_shows=20, cols=10):
        # compute embedding of merge data
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        B,_,_ = x0.shape
        dataset = x0.reshape(B,-1)
        p = reducer.fit_transform(dataset)
        fig,axs = plt.subplots()
        axs.scatter(p[:,0],p[:,1], color='blue',edgecolor='white')
        axs.set_axis_off()
        axs.set_title('reversed p')
        plt.savefig("results/reverse.umap.png")
        plt.close()


def density_plot(data):
    for i in range(data.shape[0]):
        kde = gaussian_kde(data[i,:])
        x_range = np.linspace(min(data[i,:]), max(data[i,:]),100)
        plt.plot(x_range, kde(x_range))
    # sns.kdeplot(data)
    plt.savefig("results/sample.density.png")
    plt.close()
    # step 1 : change base on N not on batch 
    # step 2 : add the original density plot as a comparsion 
    # step 3 : in the desity plot is similar then label matters
    # step 4: based on the results from step 3, adjust the model or add the label parameters as for training.