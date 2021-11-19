# ct_classification
3D CT classification using TCIA dataset

The code is based on https://github.com/kenshohara/3D-ResNets-PyTorch and has been revised to accommodate general datasets and different 3D CNN models.

Since CT does not have RGB channels, the following argument is needed `--n_input_channels 1`

The following command is used for training tcia datset

`python train.py --rootpath='/home/kaeunkim/tcia_ct_npz/ct_classification/' --model='resnet' --model_depth=50 --experiment resnet_test --result_path='/home/kaeunkim/classification/ct_classification/checkpoints/' --n_input_channels 1 --n_classes=4 --tensorboard`

## 1. Task 1
Using TCIA dataset and its clinical data, predict the T_stage of the patient (Classification).

  **Input**: 3D CT data per patient
  
  **Label**: the T_stage of the patient
  
  **Output**:
  
![image](https://user-images.githubusercontent.com/51257208/142590081-2300e485-9fc7-48a3-b5e9-81822f4e2882.png)

  **Testing/Inference**
  
  ![image](https://user-images.githubusercontent.com/51257208/142590894-de53ff85-992d-45c1-9102-37bdc612ccab.png)

  **Conclusion:**
  
  The task is too difficult as the nodule that affects the T_stage is too small compared to the entire 3D CT data. Therefore, validation accuracy was higher than training accuracy and when performing inferences, it was outputing the same result (shown above).
  
## 2. Task 2
Using TCIA dataset, its clinical dataset, and its nodule annotation data, predict the T_stage of the patient (Classification).
