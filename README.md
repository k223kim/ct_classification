# ct_classification
3D CT classification using TCIA dataset

The code is based on https://github.com/kenshohara/3D-ResNets-PyTorch and has been revised to accommodate general datasets and different 3D CNN models.

Since CT does not have RGB channels, the following argument is needed `--n_input_channels 1`

There are number of CT classification tasks performed in different approaches below.

## 1. Task 1 (./data_in_folder)
Using TCIA CT dataset and its clinical data, predict the T_stage of the patient (3D Classification, dataset structured folder-wise).

## 2. Task 2 (./data_in_csv)
Using TCIA CT dataset, its clinical dataset, and its nodule annotation data, predict the T_stage of the patient (2D Classification).
