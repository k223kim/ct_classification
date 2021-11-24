import torch
from torch import nn
import timm

model = timm.create_model('resnet50', pretrained=True, num_classes=4)

def generate_model(opt):
    possible_models = timm.list_models("*{}*".format(opt.model))
    assert len(possible_models) > 0

    model = timm.create_model(opt.model+opt.model_depth, pretrained=opt.pretrained, num_classes=opt.n_classes)
    
    if opt.n_input_channels != 3:
        #get model tree
        model_tree = []
        #e.g. model.stem.conv1
        #e.g. model.conv1
        model_dict = model.__dict__["_modules"]
        key, val = list(model_dict.items())[0]
        model_tree.append(str(key))
        while type(val) != torch.nn.modules.conv.Conv2d:
            for k, v in val.named_children():
                model_tree.append(str(k))
                val = v
                break
                
        #update the first conv layer
        new_first_conv = torch.nn.Conv2d(opt.n_input_channels, val.out_channels, val.kernel_size, val.stride, val.padding, val.dilation, val.groups, val.bias, val.padding_mode) 
        if len(model_tree) == 1:
            setattr(model, model_tree[0], new_first_conv)
        else:
            new_attr = getattr(model, model_tree[0])
            for i in range(1, len(model_tree)-1):
                new_attr2 = getattr(new_attr, model_tree[i])
                new_attr = new_attr2

            setattr(new_attr, model_tree[-1], new_first_conv)
        
    return model