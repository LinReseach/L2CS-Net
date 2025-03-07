import argparse
import numpy as np
import cv2
import time
import socket
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import pandas as pd
import numpy as np

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS

"""
                                    ----------------------------------------------------------
 The following code allow to perform gaze tracking with the L2CS model trained on Gaze360 or MPIIGaze. 
 The algorithm can be used as a real time one performing eye-tracking on real time webcam video or can be used to perform eyes 
 tracking on pre-recorded video. The latter is useful if we want to compare different models. 
 In case of video processing the video must be provided already split in frames in the directory frame/frame

"""


def parse_args():
    """Parse input arguments."""
    """model parameters"""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360 or MPIIGaze.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/models/Gaze360-20220914T091057Z-001/Gaze360/L2CSNet_gaze360.pkl', type=str)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args


def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model


def prediction(transformations, model, frame):
    cudnn.enabled = True

    batch_size = 24
    gpu = select_device(args.gpu_id, batch_size=batch_size)

    model.cuda(gpu)
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)  # 0 for gpu, -1 for CPU
    idx_tensor = [idx for idx in range(90)]  # 28 for MPIIGaze, 90 for Gaze360
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x = 0

    """ 
    # Code for real time video processing from the webcam
    #cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    #if not cap.isOpened():
    #    raise IOError("Cannot open webcam")
    """

    # timer for algorithm fps computation
    start_fps = time.time()

    """
    In case of pre recorded video processing useful to evaluate different models is worthy to perform eye tracking 
    on single person. The following code is for single person eye tracking. 
    """
    # pitch_predicted = None
    # yaw_predicted
    faces = detector(frame)
    if faces is not None:
        # variables used in case of single person eye tracking
        biggest_width = 0
        biggest_height = 0
        x_min_ = 0
        x_max_ = 0
        y_min_ = 0
        y_max_ = 0
        for box, landmarks, score in faces:
            if score < .85: #default .95
                continue
            x_min = int(box[0])
            if x_min < 0:
                x_min = 0
            y_min = int(box[1])
            if y_min < 0:
                y_min = 0
            x_max = int(box[2])
            y_max = int(box[3])
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            # The following if allow us to save just the values of the biggest square (single-person gaze tracking)
            # Demo_pepper for
            if bbox_width > biggest_width:
                biggest_width = bbox_width
                biggest_height = bbox_height
                x_min_ = x_min
                x_max_ = x_max
                y_min_ = y_min
                y_max_ = y_max
            # x_min = max(0,x_min-int(0.2*bbox_height))
            # y_min = max(0,y_min-int(0.2*bbox_width))
            # x_max = x_max+int(0.2*bbox_height)
            # y_max = y_max+int(0.2*bbox_width)
            # bbox_width = x_max - x_min
            # bbox_height = y_max - y_min

        # put inside the for loop for eye tracking of all the faces in the frame (now it's just the bigger one)
        # Crop image
        img = frame[y_min_:y_max_, x_min_:x_max_]

        # img = cv2.resize(img, (224, 224))
        try:
            img = cv2.resize(img, (224, 224))
        except Exception as e:
            return None, None, None
            print(str(e))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        img = transformations(im_pil)
        img = Variable(img).cuda(gpu)
        img = img.unsqueeze(0)

        # gaze prediction
        gaze_yaw, gaze_pitch = model(img) #yaw, pitch

        """ this can be processed all once outside the cicle"""
        pitch_predicted = softmax(gaze_pitch)
        yaw_predicted = softmax(gaze_yaw)

        # Get continuous predictions in degrees.

        # Gaze360
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180 # +7 for pitch compensation
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

        #MPIIGaze
        #pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 42 + 7 # +7 for pitch compensation
        #yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 42

        pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
        yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0
        if frame.shape == (2160, 3840, 3):
            frame = cv2.resize(frame, (1920, 1080))
            draw_gaze(x_min_/2, y_min_/2, biggest_width/2, biggest_height/2, frame, (yaw_predicted, pitch_predicted),
                  color=(0, 0, 255))
            cv2.rectangle(frame, (int(x_min_/2), int(y_min_/2)), (int(x_max_/2), int(y_max_/2)), (0, 255, 0), 1)

        draw_gaze(x_min_ , y_min_ , biggest_width , biggest_height, frame,
                  (yaw_predicted, pitch_predicted),
                  color=(0, 0, 255))
        cv2.rectangle(frame, (x_min_, y_min_), (x_max_ , y_max_ ), (0, 255, 0), 1)
        myFPS = 1.0 / (time.time() - start_fps)
        cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1,
                    cv2.LINE_AA)

        # cv2.imshow("Demo", frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #    break
        # success, frame = cap.read()
        # cap.release()
        # cv2.destroyAllWindows()
        return frame, pitch_predicted, yaw_predicted


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    # processing from local
    image_folder = 'datasets/SocialAI/headcrop7'
    # participant = 'p2'
    outputname = 'finetuning_val/'
    video_name = outputname + 'headcrop7.avi'
    output_file_name = outputname + 'headcrop7.csv'
    # Transformation needed after the face detection

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    arch = args.arch
    snapshot_path = args.snapshot

    model = getArch(arch, 90)  # 28 for MPIIGaze, 90 for Gaze360
    #model = torch.nn.DataParallel(model)  # only for MPII
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)  #, map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)

    # processing with local images
    # images = [int(img[5:-4]) for img in os.listdir(image_folder) if img.endswith(".jpg")] #remove 'frame' and order images
    #  images = [img for img in os.listdir(image_folder) if (img.endswith(".jpg") and img.startswith("00"))] 
    images = [img for img in os.listdir(image_folder) if (img.endswith(".jpg"))]  # from pepper experiment
    images = sorted(images)
    # print(images[0:15])
    # frame = cv2.imread(os.path.join(image_folder, "frame" + str(images[0]) + ".jpg"))
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    if frame.shape == (2160, 3840, 3):
        frame = cv2.resize(frame, (1920, 1080))
    # print(frame.shape)

    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 5, (width, height))

    # we save pitch and yaw into a pandas DataFrame
    pitch_predicted_ = []
    yaw_predicted_ = []
    for image in images:
        # print(image)
        # image = cv2.imread(os.path.join(image_folder, "frame" + str(image) + ".jpg"))
        # print(image[2:])
        image = cv2.imread(os.path.join(image_folder, image))
        img, pitch_predicted, yaw_predicted = prediction(transformations, model, image)
        # img = prediction(transformations, model, image)
        pitch_predicted_.append(pitch_predicted)
        yaw_predicted_.append(yaw_predicted)
        video.write(img)

    dataframe = pd.DataFrame(
        data=np.concatenate([np.array(pitch_predicted_, ndmin=2), np.array(yaw_predicted_, ndmin=2)]).T,
        columns=["yaw", "pitch"])
    dataframe.to_csv(output_file_name, index=False)
    cv2.destroyAllWindows()
    video.release()

    """
    while True:
        # frame = connect.get_img()
        # print("here 1")
        img = prediction(transformations, model, frame)
        # print("here 2")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('pepper stream', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    connect.close_connection()
    """
    print("--- Complete excecution = %s seconds ---" % (time.time() - start_time))
