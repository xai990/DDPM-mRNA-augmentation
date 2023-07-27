import torch.utils.data

def gather(const:torch.Tensor, t:torch.Tensor):
    c = const.gather(-1,t) # ( batch_size , num_steps) dim = -1 targetting on num_steps
    # c.size() should be 100 * 1
    return c.view(-1,1,1,1) # (batch_size, 1, 1 , 1)
