import os
import sys
import json
import subprocess
import numpy as np                
import torch
import time
from opts import parse_opts
from data import create_dataset
from model import generate_model
from utils import AverageMeter, Logger, calculate_accuracy, check_path

if __name__=="__main__":
    opt = parse_opts()
    opt.device = torch.device(f'cuda:{opt.device_num}' if not opt.no_cuda else 'cpu')
    if not opt.no_train:
        train_dataloader = create_dataset('train', opt)
        train_dataset_size = len(train_dataloader)
        print('The number of training cases = %d' % train_dataset_size)
    if not opt.no_val:
        val_dataloader = create_dataset('val', opt)
        val_dataset_size = len(val_dataloader)
        print('The number of validation cases = %d' % val_dataset_size)
    
    result_path = check_path(opt.result_path, opt.experiment)
        
    train_result_path = os.path.join(result_path, "train")
    os.mkdir(train_result_path)
    val_result_path = os.path.join(result_path, "val")
    os.mkdir(val_result_path)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    
    model = generate_model(opt)
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    
    log_path = check_path("./logs", opt.experiment)
        
    batch_logger = Logger(os.path.join(log_path, 'train_batch_logger.log'), ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    epoch_logger = Logger(os.path.join(log_path, 'train_epoch_logger.log'), ['epoch', 'loss', 'acc', 'lr'])

    if opt.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 0:
            tb_writer_train = SummaryWriter(log_dir=train_result_path)
            tb_writer_val = SummaryWriter(log_dir=val_result_path)
        else:
            tb_writer_train = SummaryWriter(log_dir=train_result_path, purge_step=opt.begin_epoch)
            tb_writer_val = SummaryWriter(log_dir=val_result_path, purge_step=opt.begin_epoch)
    
    for epoch in range(opt.begin_epoch, opt.n_epochs+1):
        if not opt.no_train:
            model.to(opt.device)
            model.train()

            train_batch_time = AverageMeter()
            train_data_time = AverageMeter()
            train_losses = AverageMeter()
            train_accuracies = AverageMeter()

            end_time = time.time()            
            for i, (inputs, targets) in enumerate(train_dataloader):
                train_data_time.update(time.time() - end_time)
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)
                train_losses.update(loss.item(), inputs.size(0))
                train_accuracies.update(acc, inputs.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_batch_time.update(time.time() - end_time)
                end_time = time.time()
                
                batch_logger.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * train_dataset_size + (i + 1),
                    'loss': train_losses.val,
                    'acc': train_accuracies.val,
                    'lr': optimizer.param_groups[0]['lr']
                })

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {train_batch_time.val:.3f} ({train_batch_time.avg:.3f})\t'
                      'Data {train_data_time.val:.3f} ({train_data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                          epoch, i + 1, train_dataset_size, train_batch_time=train_batch_time,
                          train_data_time=train_data_time, loss=train_losses, acc=train_accuracies))

            epoch_logger.log({
                'epoch': epoch,
                'loss': train_losses.avg,
                'acc': train_accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })        
        if not opt.no_val:
            print('validation at epoch {}'.format(epoch))
            model.to(opt.device)
            model.eval()

            val_batch_time = AverageMeter()
            val_data_time = AverageMeter()
            val_losses = AverageMeter()
            val_accuracies = AverageMeter()

            end_time = time.time()

            with torch.no_grad():
                for i, (inputs, targets) in enumerate(val_dataloader):
                    val_data_time.update(time.time() - end_time)
                    inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                    targets = targets.to(opt.device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    acc = calculate_accuracy(outputs, targets)

                    val_losses.update(loss.item(), inputs.size(0))
                    val_accuracies.update(acc, inputs.size(0))

                    val_batch_time.update(time.time() - end_time)
                    end_time = time.time()

                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {val_batch_time.val:.3f} ({val_batch_time.avg:.3f})\t'
                          'Data {val_data_time.val:.3f} ({val_data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                              epoch,
                              i + 1,
                              len(val_dataloader),
                              val_batch_time=val_batch_time,
                              val_data_time=val_data_time,
                              loss=val_losses,
                              acc=val_accuracies))
            
            
        if opt.tensorboard is not None:
            tb_writer_train.add_scalar('loss/epoch', train_losses.avg, epoch)
            tb_writer_train.add_scalar('accuracy/epoch', train_accuracies.avg, epoch)   
            tb_writer_val.add_scalar('loss/epoch', val_losses.avg, epoch)
            tb_writer_val.add_scalar('accuracy/epoch', val_accuracies.avg, epoch)   

        if epoch % opt.checkpoint == 0:
            save_file_path = os.path.join(result_path, 'save_{}_{}.pth'.format(opt.model, epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
            torch.save(states, save_file_path)    