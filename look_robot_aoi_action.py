import argparse
import random

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

from numpy.linalg import inv


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

    def adjust_head(self, pitch, yaw):
        self.s.sendall(bytes("head {:0.3f} {:0.3f}".format(pitch, yaw).encode()))

    def say(self, text):
        self.s.sendall(bytes(f"say {text}".encode()))

    def nod(self):
        self.s.sendall(bytes("nod".encode()))

    def enable_tracking(self):
        self.s.sendall(bytes("track True".encode()))

    def disable_tracking(self):
        self.s.sendall(bytes("track False".encode()))

    def idle(self):
        self.s.sendall(bytes("idle".encode()))

    def look(self, x, y):
        self.s.sendall(bytes("look;{:0.5f};{:0.5f}".format(x, y).encode()))

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
            pitch_predicted = softmax(gaze_pitch) # this is not pitch but its actually yaw. We are leaving it as it is for coherence with other analysis
            yaw_predicted = softmax(gaze_yaw) # # this is not yaw but its actually pitch. We are leaving it as it is for coherence with other analysis

            # Get continuous predictions in degrees.
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180 # compensation (remembner that this ios actually pitch)

            pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
            yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

            draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (yaw_predicted, pitch_predicted),
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

def get_ladybug_to_eye_matrix(dir_eyes):
    """Creates a transformation matrix from the eye coordinate system."""
    up_vector = np.array([0, 0, 1], np.float32)
    z_axis = dir_eyes.flatten()
    x_axis = np.cross(up_vector, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=0)

def transform(g_p, pos, h_eye_cam, r_left):
    """Transforms gaze direction into the robot's coordinate system."""
    eye_pos = np.array([-100, -r_left, h_eye_cam]) if pos == 2 else np.array([0, 0, 0])  # Adjust as needed

    # Direction of eyes
    dir_eyes = eye_pos / np.linalg.norm(eye_pos)
    
    # Compute gaze coordinate system
    gaze_cs = get_ladybug_to_eye_matrix(dir_eyes)
    gaze_dir_lb = np.matmul(inv(gaze_cs), g_p.T)
    
    # Scaling factor for projection onto robot screen plane
    k = (d_horizontal_robot_screen - eye_pos[0]) / gaze_dir_lb[0]
    target = (gaze_dir_lb.T * k) + eye_pos
    
    # Adjust target coordinates for robot's screen plane
    target[:, 0] = d_horizontal_robot_screen - target[:, 0]
    target[:, 1] = -target[:, 1]
    target[:, 2] = target[:, 2] - d_vertical_robot_screen
    
    # Output as DataFrame with named columns
    return pd.DataFrame(target, columns=['virtual2d_x', 'virtual2d_y', 'depth'])

def find_aoi(x, y):
    # Loop through each AOI and check if the point is within the AOI's boundary
    for i, (center, width, height) in enumerate(zip(rect_centers, rect_width, rect_height)):
        x_center, y_center = center
        half_width = width / 2
        half_height = height / 2

        # Define boundaries for the current AOI
        x_min = x_center - half_width
        x_max = x_center + half_width
        y_min = y_center - half_height
        y_max = y_center + half_height

        # Check if the point is within the boundaries
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return f"AOI {i + 1}"

    # If the point is not in any AOI
    return "elsewhere"



if __name__ == '__main__':
    args = parse_args()

    # create a connection with pepper
    connect = socket_connection(ip=args.ip, port=args.port, camera=args.cam_id)

    video_name = 'pepper_example.avi'

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


    pitch_predicted_ = []
    yaw_predicted_ = []
 
    frame = connect.get_img()
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 5, (width, height))
    eye_height = 1.2  # Example value for eye height
    camera_height = 1.2
    h_eye_cam = eye_height - camera_height
    # Parameters for transformation
    d_horizontal_robot_screen = 0
    d_vertical_robot_screen = 0
    r_left = 0  # Adjust as needed
    pos = 2  # Example position input; adjust as necessary
    # Define AOI centers, widths, and heights
    # rect_centers = [(0, -32.0)]
    # rect_width = [50]
    # rect_height = [80]
    rect_centers = [(0, -8.1),(0,-28),(-30.8,-32), (29.7,-30.8)]#tablet_original(0,-32)
    rect_width = [80 * 2 / 9.19+2, 24.6+5, 150 * 2 / 9.19, 150 * 2 / 9.19]
    rect_height = [80 * 2 / 9.19+2, 17.5+5, 150 * 2 / 9.19, 150 * 2 / 9.19]
	#pre-definition

    connect.enable_tracking()
    while True:
        frame = connect.get_img()
        img, pitch, yaw = prediction(transformations, model, frame)

        if (pitch, yaw) != (None, None):
            x = np.cos(pitch) * np.sin(yaw)
            y = np.sin(pitch)
            z = -np.cos(yaw) * np.cos(pitch)
            g_p = np.array([[x, y, z]])
            dfn = transform(g_p, pos, h_eye_cam, r_left)[['virtual2d_x', 'virtual2d_y', 'depth']]

            output_x = dfn['virtual2d_y'].values[0]
            output_y = dfn['depth'].values[0]
            print(output_x, output_y)
            print(find_aoi(output_x, output_y))

            if find_aoi(output_x, output_y)=='AOI 1':
                connect.say("hello!")
            elif find_aoi(output_x, output_y)=='AOI 2':
                connect.say("my tablet?")
                time.sleep(1)
                connect.adjust_head(0.3,0)
                time.sleep(2)
                # connect.adjust_head(-0.3,0)
                # time.sleep(0.5)
            elif find_aoi(output_x, output_y) == 'AOI 3':

                connect.say("my left arm?")
                time.sleep(1)
                connect.adjust_head(-0.3,-0.6)
                time.sleep(2)
                # connect.adjust_head(-0.3,-0.3)
                # time.sleep(0.5)
            elif find_aoi(output_x, output_y) == 'AOI 4':

                connect.say("my right arm?")
                time.sleep(1)
                connect.adjust_head(0.3,0.3)
                time.sleep(3)
                #connect.adjust_head(-0.3,0.3)
                # time.sleep(0.5)
            else:
                print('elsewhere')
                # pass
                #connect.say("elsewhere!")
       


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        #print(pitch, yaw)

        # if (pitch,yaw) != (None, None):
        #     pitch = pitch * 180 / np.pi
        #     yaw = yaw * 180 / np.pi
        #     if (-5 < yaw < 5) & (-10 <pitch < 10):
        #         connect.say("I kun")

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


