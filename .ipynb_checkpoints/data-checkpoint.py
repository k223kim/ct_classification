import torch
import numpy as np
import trans

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
    def __init__(self, paths, phase):
        self.paths = paths
        self.phase = phase
    
    def __getitem__(self, index):
        path = self.paths[index]
        images = np.load(path)['data']
        transforms = get_transform()
        images_transformed = transforms(images)
        
        label = path.split('/')[-2]
        
        return images_transformed, label
        
    def __len__(self):
        return len(self.paths)

def get_transform(convert=True):
    #Later on, can take opt so that it can have multiple transformations
    """
    creates a list of transformation 
    
    """
    transform_list = []
    if convert:
        transform_list += [trans.ToTensor()]
        transform_list += [trans.Normalize(mean, std)] #mean and std can be fixed later on
        
    return trans.Compose(transform_list)        