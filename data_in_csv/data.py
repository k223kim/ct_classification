import torch
import numpy as np
import trans
import glob
import os
import 

class TCIACSVDataset(torch.utils.data.Dataset):
    """
    dataset class for the TCIA dataset
    
    Attributes
    ----------
    dicts : 
        list of dictionaries for each instance
        Example:
                {'patient': 'patient_num',
                 'patient_path': 'root_path',
                 'mode': 'CT',
                 'dcm_path': '/../../.dcm',
                 'annotation_path': '/../../.xml',
                 'label': 2,
                 'manufacturer': 'manufacturer',
                 'slice_thickness': 10.0,
                 'bbox': '[x1, y1, x2, y2]',
                 'bbox_area': (x2-x1) * (y2-y1),
                 'split1': 'train'}
    
    transform : object
        list of albumentation transforms
    """
    def __init__(self, dicts, opt):
        self.dicts = dicts    
        self.opt = opt
    
    def __getitem__(self, index):
        dcm_path = self.dicts[index]['dcm_path']
        bbox_path = self.dicts[index]['annotation_path']
        
        #get dicom image
        dcm = pydicom.read_file(dcm_path, force=True)
        dcm_img = dcm.pixel_array
        
        #get annotation
        tree = ElementTree.parse(bbox_path)
        root = tree.getroot()
        bounding_boxes = []
        size_tree = root.find('size')
        width = float(size_tree.find('width').text)
        height = float(size_tree.find('height').text)
        for object_tree in root.findall('object'):
            for bounding_box in object_tree.iter('bndbox'):
                xmin = float(bounding_box.find('xmin').text)
                ymin = float(bounding_box.find('ymin').text)
                xmax = float(bounding_box.find('xmax').text)
                ymax = float(bounding_box.find('ymax').text)
            bounding_box = [xmin, ymin, xmax, ymax]
            bounding_boxes.append(bounding_box)
        if len(bounding_boxes) > 1:
            area = []
            for b in bounding_boxes:
                x = abs(b[2] - b[0])
                y = abs(b[3] - b[1])
                this_area = x*y
                area.append(y)
            area = np.array(area)
            max_idx = area.argmax()
            max_bbox = bounding_boxes[max_idx]
        else:
            max_bbox = bounding_boxes[0]
            
        #normalize to uint8
        norm_img = np.uint8(((dcm_img - dcm_img.min()) / (dcm_img.max() - dcm_img.min())) * 255.0)            
        
        #crop image
        cropped_img = norm_img[int(max_bbox[1]):int(max_bbox[3]), int(max_bbox[0]):int(max_bbox[2])]
        
        #resize
        res = cv2.resize(cropped_img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        
        transform = get_transform(opt)
        if transform is not None:
            image = transform(image=res)['image']
        else:
            image = res
            
        label = self.dicts[index]['label']  
        
        return image, int(label)-1
        
    def __len__(self):
        return len(self.dicts)      

def make_datapath_list(phase, opt):
    """
    creates a list that has json dictionaries of each instance
    
    Parameters 
    ----------
    phase : 'train' or 'val'
    
    Parameters 
    ----------
    path_list : list
    """
    data = pd.read_csv(opt.csvpath)
    json_str = data[data["split1"] == phase].to_json(orient="records")
    json_dict = json.loads(json_str)
    
    return json_dict

def get_transform(opt):
    #Later on, can take opt so that it can have multiple transformations
    #future work: add more augmentation methods to this list based on the opt variables
    """
    creates a list of transformation using albumentations
    
    """
    transform_list = []
    if opt.convert:
        transform_list += ToTensorV2()
        if opt.n_input_channels == 1:
            transform_list += A.Normalize(mean=opt.mean, std=opt.std)
        else:
            new_mean = np.repeat(opt.mean, opt.n_input_channels).tolist()
            new_std = np.repeat(opt.std, opt.n_input_channels).tolist()
            transform_list += A.Normalize(mean=new_mean, std=new_std)
        
    return A.Compose(transform_list)    

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
    dataset = TCIACSVDataset(data_path, opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    return dataloader