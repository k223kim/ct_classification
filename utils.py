import csv
import torch
import os
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, "w")
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size
    
def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)

def check_path(path, experiment):
    new_path = os.path.join(path, experiment)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
        return new_path
    else:
        #this means that the experiment name is overlapping
        timestr = time.strftime("%Y%m%d-%H%M%S")
        new_path = os.path.join(path, experiment+"_"+ timestr)
        os.mkdir(new_path)
        return new_path