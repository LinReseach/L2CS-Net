# File Explanation

## Dataset and Model Files
- **datasets.py**: Creates the dataset object for training. Defines the SocialAI dataset class.
- **model.py**: Defines the L2CS model.

## Training and Testing
- **train.py**: Code for training the model, including fine-tuning.
- **test.py**: Tests the trained model.
- **leave_one_out_eval.py**: Implements leave-one-out validation.
- **utils.py**: Contains utility functions used across different files.

## Demos
- **demo.py**: Original L2CS demo from GitHub.
- **demo_local.py**: Runs L2CS prediction on given images.
- **demo_pepper.py**: Real-time mirror experiment on Pepper.
- **demo_ft_lin.py**: Uses a fine-tuned L2CS model for gaze prediction.
- **demo_local_folder.py**: Runs L2CS demo on a folder of images.
- **demo_pepper_lin.py**: Gaze-following real-time experiment with modified code.
- **demo_pepper2.py**: Second version of the real-time gaze-following experiment.

## Gaze Detection Experiments
- **look_robot_aoi_action.py**: Detects which AOI of the robot body the user is looking at and triggers an action.
- **look_robot_aoi.py**: Detects which AOI of the robot body the user is looking at and announces the name.
- **look_robot_or_not.py**: Detects if the user is looking at the robot.

## Custom Training and Testing Scripts
- **train_local_lin.py**: Custom training script.
- **train_local_lin_newdata.py**: Training script with new dataset.
- **train_local_lin_newdata_l.py**: Another variant of training with new data.
- **test_local_lin.py**: Custom test script.
- **epoch_result/**: Stores training results.

## Dataset Variants
- **dataset_local_lin.py**: Uses a new dataset.
- **datasets_local_no_par1315.py**: Uses a dataset excluding participants 13 and 15.

# Commands to Run L2CS Pretrained Models

demo with l2cs trained on Gaze360:
python3 demo_local.py --snapshot models/Gaze360-20220914T091057Z-001/Gaze360/L2CSNet_gaze360.pkl


NOTE: when you train with MPII remember to change number of bins and pitch yaw prediction transformation
demo with l2cs trained on MPIIGaze:
python3 demo_local.py --snapshot models/MPIIGaze-20220914T091058Z-001/MPIIGaze/fold1.pkl


demo on pepper streming data with l2cs trained on Gaze360:
python3 demo_pepper.py --ip=10.15.3.25 --port=12345 --cam_id=4 --snapshot models/Gaze360-20220914T091057Z-001/Gaze360/L2CSNet_gaze360.pkl

demo on pepper streming data with l2cs trained on MPIIGaze:
python3 demo_pepper.py --ip=192.168.0.167 --port=12345 --cam_id=4 --snapshot models/MPIIGaze-20220914T091058Z-001/MPIIGaze/fold1.pkl


GPU - CPU changes: map_location, cuda, gpu_id, .cpu, batch_size

--------    how to do fine tuning    ----------
 python train_local.py --dataset socialai 

--------    time of execution    ----------
[video_1, MPIIGaze, CPU] = 52.0 s

[video_1, MPIIGaze, GPU, batch_size 16] = 44.7

[video_1, MPIIGaze, GPU, batch_size 32] = 44.5

# Fine-tuning: Dataset Composition
The models were trained using two datasets:

Low-quality images only

Mixed dataset with both 4K and low-quality images

Train, Validation, and Test Sets

Training Set: Participants 1,2,4,5,7,8,9,14,15,18,19,20,21.

Validation Set: Participants 12,13.

Test Set: Participants 3,6,10,11,16,17 (only low-quality images).

Since the robot uses its built-in monocular camera at deployment, only low-quality images are used in the test set to ensure realistic performance estimation.

Model Updates

Freeze the entire network except for the last two fully connected layers.

Freeze the first convolutional layer and train the last two fully connected layers along with four central convolutional layers.

Regularization

Dropout layers added after each convolutional layer to prevent overfitting.

This README provides an overview of the repository's structure, key files, commands, dataset composition, and fine-tuning strategies.


