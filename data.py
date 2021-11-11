import torch
import numpy as np
import trans
import glob
import os

class TCIADataset(torch.utils.data.Dataset):
    """
    dataset class for the TCIA dataset
    
    Attributes
    ----------
    file_list : list 
        list of data paths
    
    transform : object
        instance of a preprocessing class
    """
    def __init__(self, paths, opt):
        self.paths = paths
        self.opt = opt
    
    def __getitem__(self, index):
        path = self.paths[index]
        images = np.load(path)['data']
        transforms = get_transform(self.opt)
        images_transformed = transforms(images)
        images_transformed = torch.unsqueeze(images_transformed, 0)
        permute_image = images_transformed.permute(0, 3, 1, 2)
        label = int(path.split('/')[-2]) - 1
        
        return permute_image, label
        
    def __len__(self):
        return len(self.paths)

def get_transform(opt):
    #Later on, can take opt so that it can have multiple transformations
    #future work: add more augmentation methods to this list based on the opt variables
    """
    creates a list of transformation 
    
    """
    transform_list = []
    if opt.convert:
        transform_list += [trans.ToTensor()]
        transform_list += [trans.Normalize(opt.mean, opt.std)]
        
    return trans.Compose(transform_list)        

def make_datapath_list(phase, opt):
    """
    creates a list that saves all the datapath
    
    Parameters 
    ----------
    phase : 'train' or 'val'
    
    Parameters 
    ----------
    path_list : list
    """
    path_list = [g for g in glob.glob(os.path.join(opt.rootpath, phase, '*', '*'))]
    
    return path_list

def create_dataset(phase, opt):
    """
    creates a dataloader for the particular dataset
    
    Parameters 
    ----------
    phase : 'train' or 'val'
    
    Parameters 
    ----------
    dataloader : torch.utils.data.DataLoader
    """    
    data_path = make_datapath_list(phase, opt)
    dataset = TCIADataset(data_path, opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    return dataloader