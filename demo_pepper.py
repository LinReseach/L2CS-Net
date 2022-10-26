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
    
    
class socket_connection():
    """
    Class for creating socket connection and retrieving images
    """
    def __init__(self, ip, port, camera, **kwargs):
        """
        Init of vars and creating socket connection object.
        Based on user input a different camera can be selected.
        1: Stereo camera 1280*360
        2: Stereo camera 2560*720
        3: Mono camera 320*240
        4: Mono camera 640*480
        """
        # Camera selection
        if camera == 1:
            self.size = 1382400  # RGB
            self.size = 921600  # YUV422
            self.width = 1280
            self.height = 360
            self.cam_id = 3
            self.res_id = 14
        elif camera == 2:
            self.size = 5529600  # RGB
            self.size = 3686400  # YUV422
            self.width = 2560
            self.height = 720
            self.cam_id = 3
            self.res_id = 13
        elif camera == 3:
            self.size = 230400 # RGB
            self.size = 153600 # YUV422
            self.width = 320
            self.height = 240
            self.cam_id = 0
            self.res_id = 1
        elif camera == 4:
            self.size = 921600 # RGB
            self.size = 614400 # YUV422
            self.width = 640
            self.height = 480
            self.cam_id = 0
            self.res_id = 2
        else:
            print(f"Invalid camera selected... choose between 1 and 4, got {camera}")
            exit(1)

        self.COLOR_ID = 13
        self.ip = ip
        self.port = port

        # Initialize socket socket connection
        self.s = socket.socket()
        try:
            self.s.connect((self.ip, self.port))
            print("Successfully connected with {}:{}".format(self.ip, self.port))
        except:
            print("ERR: Failed to connect with {}:{}".format(self.ip, self.port))
            exit(1)


    # def get_img(self):
    #     """
    #     Send signal to pepper to recieve image data, and convert to image data
    #     """
    #     self.s.send(b'getImg')
    #     pepper_img = b""
    #
    #     l = self.s.recv(self.size - len(pepper_img))
    #     while len(pepper_img) < self.size:
    #         pepper_img += l
    #         l = self.s.recv(self.size - len(pepper_img))
    #
    #     im = Image.frombytes("RGB", (self.width, self.height), pepper_img)
    #     cv_image = cv2.cvtColor(np.asarray(im, dtype=np.uint8), cv2.COLOR_BGRA2RGB)
    #
    #     return cv_image[:, :, ::-1]

    def say(self, text):
        self.s.sendall(bytes(f"say {text}".encode()))

    def get_img(self):
        #     """
        #     Send signal to pepper to recieve image data, and convert to image data
        #     """
        self.s.send(b'getImg')
        pepper_img = b""

        l = self.s.recv(self.size - len(pepper_img))
        while len(pepper_img) < self.size:
            pepper_img += l
            l = self.s.recv(self.size - len(pepper_img))

        arr = np.frombuffer(pepper_img, dtype=np.uint8)
        y = arr[0::2]
        u = arr[1::4]
        v = arr[3::4]
        yuv = np.ones((len(y)) * 3, dtype=np.uint8)
        yuv[::3] = y
        yuv[1::6] = u
        yuv[2::6] = v
        yuv[4::6] = u
        yuv[5::6] = v
        yuv = np.reshape(yuv, (self.height, self.width, 3))
        image = Image.fromarray(yuv, 'YCbCr').convert('RGB')
        image = np.array(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return image




def parse_args():
    """Parse input arguments."""

    """model parameters"""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    """connection parameters"""
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. Default 127.0.0.1")
    parser.add_argument("--port", type=int, default=12345,
                        help="Pepper port number. Default 9559.")
    parser.add_argument("--cam_id", type=int, default=3,
                        help="Camera id according to pepper docs. Use 3 for "
                             "stereo camera and 0. Default is 3.")
    args = parser.parse_args()

    return args


def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model


def prediction(transformations, model, frame):
    cudnn.enabled = True

    # arch = args.arch
    batch_size = 16
    #cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    # snapshot_path = args.snapshot

    model.cuda(gpu)
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)  # 0 for gpu, -1 for CPU
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x = 0

    pitch_predicted = None
    yaw_predicted = None

    #cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    #if not cap.isOpened():
    #    raise IOError("Cannot open webcam")

    start_fps = time.time()

    #faces = detector(frame)
    faces = detector(frame)
    if faces is not None:
        for box, landmarks, score in faces:
            if score < .95:
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

            # x_min = max(0,x_min-int(0.2*bbox_height))
            # y_min = max(0,y_min-int(0.2*bbox_width))
            # x_max = x_max+int(0.2*bbox_height)
            # y_max = y_max+int(0.2*bbox_width)
            # bbox_width = x_max - x_min
            # bbox_height = y_max - y_min

            #put inside th for loop for eye tracking of all the faces in the frame (now it's just the bigger)
            # Crop image
            img = frame[y_min:y_max, x_min:x_max]
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            img = transformations(im_pil)
            img = Variable(img).cuda(gpu)
            img = img.unsqueeze(0)

            # gaze prediction
            gaze_yaw, gaze_pitch = model(img)

            """ this can be processed all once outside the cicle"""
            pitch_predicted = softmax(gaze_pitch)
            yaw_predicted = softmax(gaze_yaw)

            # Get continuous predictions in degrees.
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180 + 7  # for pitch compensation
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

            pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
            yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

            draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (pitch_predicted, yaw_predicted),
                      color=(0, 0, 255))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        myFPS = 1.0 / (time.time() - start_fps)
        cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1,
                        cv2.LINE_AA)

            #cv2.imshow("Demo", frame)
            #if cv2.waitKey(1) & 0xFF == 27:
            #    break
            #success, frame = cap.read()
            #cap.release()
            #cv2.destroyAllWindows()
        return frame, pitch_predicted, yaw_predicted


if __name__ == '__main__':
    args = parse_args()

    # create a connection with pepper
    connect = socket_connection(ip=args.ip, port=args.port, camera=args.cam_id)

    # processing from local images
    image_folder = 'frame/frame'
    video_name = 'processed_video.avi'

    """Set up parameter for the prediction"""
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

    model = getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path) #, map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict, strict=False)
    # print("here 0")

    """
    # processing with local images
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = sorted(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 5, (width, height))
    
    
    # we save pitch and yaw into a pandas DataFrame
    pitch_predicted_ = []
    yaw_predicted_ = []
    for image in images:
        image = cv2.imread(os.path.join(image_folder, image))
        img, pitch_predicted, yaw_predicted = prediction(transformations, model, image)

        pitch_predicted_.append(pitch_predicted)
        yaw_predicted_.append(yaw_predicted)
        video.write(img)

    dataframe = pd.DataFrame(data=np.concatenate([np.array(pitch_predicted_, ndmin=2), np.array(yaw_predicted_, ndmin=2)]).T, columns=["pitch", "yaw"])
    dataframe.to_csv('out.csv', index=False)
    cv2.destroyAllWindows()
    video.release()
    
    """

    pitch_predicted_ = []
    yaw_predicted_ = []
    while True:
        frame = connect.get_img()
        img, pitch, yaw = prediction(transformations, model, frame)
        # img = prediction(transformations, model, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        #print(pitch, yaw)

        if (pitch,yaw) != (None, None):
            pitch = pitch * 180 / np.pi
            yaw = yaw * 180 / np.pi
            if (-5 < yaw < 5) & (-10 <pitch < 10):
                connect.say("I kun")

        """    
        pitch_predicted_.append(pitch_predicted)
        yaw_predicted_.append(yaw_predicted)"""
        cv2.imshow('pepper stream', img)
        cv2.waitKey(1)
    """
    dataframe = pd.DataFrame(
        data=np.concatenate([np.array(pitch_predicted_, ndmin=2), np.array(yaw_predicted_, ndmin=2)]).T,
        columns=["pitch", "yaw"])
    dataframe.to_csv('out.csv', index=False)
    """
    cv2.destroyAllWindows()

    connect.close_connection()


