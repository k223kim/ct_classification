## 1. Task 1
Using TCIA dataset and its clinical data, predict the T_stage of the patient (Classification).

  **Dataset**: The CT dicom images were stacked per patient and that stack was converted to .npz. The `TCIADataset` class expects the following file structure.For the `--rootpath` argument, the path to the root of the dataset (in this case, '/home/kaeunkim/tcia_ct_npz/ct_classification/') that contains the train, val, and test folders is expected.
```
.
├── ct_classification
│   ├── train
│   │   ├── t_stage_1
│   │   │   ├── patient1.npz
│   │   │   ├── patient2.npz
│   │   │   ├── ...
│   │   ├── t_stage_2
│   │   │   ├── patient3.npz
│   │   │   ├── patient4.npz
│   │   │   ├── ...
│   │   ├── ...
│   ├── val
│   ├── test
```

  **Input**: 3D CT data per patient
  
  **Label**: the T_stage of the patient
  
  **Train** : The following command is used for training for this task
  
`python train.py --rootpath='/home/kaeunkim/tcia_ct_npz/ct_classification/' --model='resnet' --model_depth=50 --experiment resnet_test --result_path='/home/kaeunkim/classification/ct_classification/data_in_folder/checkpoints/' --n_input_channels 1 --n_classes=4 --tensorboard`
  
  **Output**:
  
![image](https://user-images.githubusercontent.com/51257208/142590081-2300e485-9fc7-48a3-b5e9-81822f4e2882.png)

  **Testing/Inference**
  
  ![image](https://user-images.githubusercontent.com/51257208/142590894-de53ff85-992d-45c1-9102-37bdc612ccab.png)

  **Conclusion:**
  
  The task is too difficult as the nodule that affects the T_stage is too small compared to the entire 3D CT data. Therefore, validation accuracy was higher than training accuracy and when performing inferences, it was outputing the same result (shown above).
