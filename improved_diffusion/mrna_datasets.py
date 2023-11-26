import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd 
import numpy as np 
from . import logger
import os 
import numpy as np  
#import torch as th 

def load_data(
    *, data_dir, batch_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (features, kwargs) pairs.

    Each features is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_files_recursively(data_dir)
    logger.debug(f"all files include:{all_files}")
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = CustomGeneDataset(all_files[0],
                                all_files[1],
                                transform= GeneDataTransform(),
                                scaler=True,
                                random_selection =GeneRandom(seed=12345),
    )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["txt"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_files_recursively(full_path))
    return results


class CustomGeneDataset(Dataset): 
    def __init__(self, genepath="data.txt", labelpath="label.txt", transform=None, scaler = None, target_transform=None, random_selection = None):
        assert os.path.exists(genepath), "gene path: {} does not exist.".format(genepath)
        logger.info(f"reading input data from {os.path.basename(genepath)}") 
        # read the gene expression 
        df = pd.read_csv(genepath, sep='\t', index_col=0)
        logger.info(f"loaded input data has {df.shape[1]} genes, {df.shape[0]} samples")
        gene_features = df.values
    
        assert os.path.exists(labelpath), "gene label path: {} does not exist.".format(labelpath)
        # read the gene labels
        df_labels, _ = pd.read_csv(labelpath, sep='\t', header=None)
        #if (df.index != df_labels.index).any():
        #     print("warining: data and labels are not ordered the same, re-ordering labels")
        #     df_labels = df_labels.loc[df.index] 
        # label = df_labels.values
        self.gene = gene_features
        #self.label = torch.from_numpy(label)
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.scaler = scaler 
        self.local_classes = None 
        self.random_selection = random_selection
        
    def __len__(self):
        return len(self.gene)
    
    def __getitem__(self,idx):
        
        #gene, label = self.gene[idx], self.label[idx]
        gene = self.gene
        out_dict = {}
        if self.transform:
            gene = self.transform(gene, self.scaler)
            
        if self.target_transform:
            label = self.target_transform(label)
        
        if self.random_selection:
            gene = self.random_selection(gene)

        return np.array(gene[idx], dtype=np.float32), out_dict



class GeneDataTransform():
        
    def __call__(self, sample, scaler= None):
        # filter the nan value to min values
        mask_sample = sample[~np.isnan(sample)]
        mask_nan = np.amin(mask_sample)
        sample = np.nan_to_num(sample, nan = mask_nan)
        
        if scaler:
            mins, maxs = np.amin(sample, axis=0)[0], np.amax(sample, axis=0)[0]        
            scaler_ = np.maximum(np.abs(mins),np.abs(maxs))
            sample = np.divide(sample, scaler_)
        sample  = sample[:,np.newaxis,:]
        #return th.from_numpy(sample).float()
        return sample


class GeneRandom():
    def __init__(self, seed=None,features = 100):
        self.seed = seed 
        self.reset_random_seeds()
        self.features = features
    def reset_random_seeds(self):
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def __call__(self, sample):
        # random select the gene 
        idx = np.random.randint(0,sample.shape[-1], self.features)       
        return sample[:,:,idx]