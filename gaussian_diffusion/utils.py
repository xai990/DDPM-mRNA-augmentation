import torch.utils.data
import argparse 

import gaussian_diffusion as gd

def gather(const:torch.Tensor, t:torch.Tensor):
    c = const.gather(-1,t) # ( batch_size , num_steps) dim = -1 targetting on num_steps
    # c.size() should be 100 * 1
    return c.view(-1,1,1,1) # (batch_size, 1, 1 , 1)


def model_and_diffusion_defaults():
    """ default for DF model"""
    return dict()


def create_model_and_diffusion():
    model = create_model()
    diffusion = create_gaussian_diffusion()

    return model, diffusion


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_learned_sigma=False,
):
    beta = gd.get_named_beta_schedule(noise_schedule,steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    

def add_dict_to_argparser(parser, default_dict):
    for k , v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v,bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)



def args_to_dict(args, keys):
    return {k:getattr(args,k) for k in keys}


def str2bool(v):
    if isinstance(v,bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
        