# ct_classification
3D CT classification using TCIA dataset

The code is based on https://github.com/kenshohara/3D-ResNets-PyTorch and has been revised to accommodate general datasets and different 3D CNN models.

Since CT does not have RGB channels, the following argument is needed `--n_input_channels 1`

The following command is used for training tcia datset

`python train.py --rootpath='/home/kaeunkim/tcia_ct_npz/ct_classification/' --model='resnet' --model_depth=50 --experiment resnet_test --result_path='/home/kaeunkim/classification/ct_classification/checkpoints/' --n_input_channels 1 --n_classes=4 --tensorboard`
