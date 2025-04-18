import os
import random

import numpy as np
import cv2

from utils import augmentation
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


class Gaze360(Dataset):
    def __init__(self, path, root, transform, angle, binwidth, train=True):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        if train==False:
          angle=90
        self.binwidth=binwidth
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)
        # img=torch.from_numpy(fimg).type(torch.FloatTensor)

        if self.transform:
            img = self.transform(img)        


        # Bin values
        bins = np.array(range(-1*self.angle, self.angle, self.binwidth))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])
        return img, labels, cont_labels, name



class Mpiigaze(Dataset): 
  def __init__(self, pathorg, root, transform, train, angle,fold=0):
    self.transform = transform
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    path=pathorg.copy()
    if train==True:
      path.pop(fold)
    else:
      path=path[fold]
    if isinstance(path, list):
        for i in path:
            with open(i) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= 42 and abs((label[1]*180/np.pi)) <= 42:
                self.lines.append(line)
   
    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))
        
  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[3]
    gaze2d = line[7]
    head2d = line[8]
    lefteye = line[1]
    righteye = line[2]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)


    pitch = label[0]* 180 / np.pi
    yaw = label[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))

    # fimg = cv2.imread(os.path.join(self.root, face))
    # fimg = cv2.resize(fimg, (448, 448))/255.0
    # fimg = fimg.transpose(2, 0, 1)
    # img=torch.from_numpy(fimg).type(torch.FloatTensor)
    
    if self.transform:
        img = self.transform(img)        
    
    # Bin values
    bins = np.array(range(-42, 42,3))
    binned_pose = np.digitize([pitch, yaw], bins) - 1

    labels = binned_pose
    cont_labels = torch.FloatTensor([pitch, yaw])


    return img, labels, cont_labels, name


# annotation columns for socialAI dataset: 'dotNr','corrResp', 'fName',
#        'sampTime', 'person_ID', 'height', 'glasses', 'pos_number',
#        'yaw','pitch', 'line', 'path', 'Train'

# root: datasets/SocialAI/
# facepaths: headcrop2/image_name

class SocialAI(Dataset):
    def __init__(self, transform, binwidth = 3 , high_res = False, train=True, training_val = True, distances = [1,2,3]):
        self.transform = transform
        self.root = "datasets/SocialAI/"
        self.training_val = str(training_val)
        self.orig_list_len = 0
        self.train = str(train)
        self.distances = distances # 1,2,3 need to be an array
        self.high_res = str(high_res)
        # self.angle = angle
        self.binwidth = binwidth
        self.lines = []
        # path = pathorg.copy()
        # if train==True:
        #     path.pop(fold)
        # else:
        #     path=path[fold]

        if self.train == "True":
            with open("datasets/SocialAI/annotation_train_no1315.csv") as f:
                lines = f.read().splitlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    # if line.split(",")[-2] == self.train and line.split(",")[3] == self.high_res and int(line.split(",")[-1]) in self.rows:
                    high_res_line = line.split(",")[1]
                    if self.high_res == "True":
                        high_res_line = "True"
                    if line.split(",")[-1] != self.training_val and int(line.split(",")[-3]) in self.distances and high_res_line == self.high_res:
                        if abs(np.double(line.split(",")[-4])*180/np.pi) <= 60 and abs(np.double(line.split(",")[-5])*180/np.pi) <= 60:
                          self.lines.append(line)
                print(len(self.lines))
        else:
            with open("datasets/SocialAI/annotation_test.csv") as f:
                lines = f.read().splitlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    # if line.split(",")[-2] == self.train and line.split(",")[3] == self.high_res and int(line.split(",")[-1]) in self.rows:
                    if int(line.split(",")[-2]) in self.distances:
                        if abs(np.double(line.split(",")[-3])*180/np.pi) <= 60 and abs(np.double(line.split(",")[-4])*180/np.pi) <= 60:
                          self.lines.append(line)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        var = True
        if self.train == "True":
            while var:
                line = self.lines[idx]

                pitch = np.double(line.split(",")[-4])
                yaw = np.double(line.split(",")[-5])

                face_path = line.split(",")[-2]
                try:
                    img = Image.open(os.path.join(self.root, face_path))
                    var = False
                except:
                    idx = idx - 1
        else:
            while var:
                line = self.lines[idx]

                pitch = np.double(line.split(",")[-3])
                yaw = np.double(line.split(",")[-4])

                face_path = line.split(",")[-1]
                try:
                    img = Image.open(os.path.join(self.root, face_path))
                    var = False
                except:
                    idx = idx - 1

        if self.transform:
            img = self.transform(img)

            # label = [np.double(el) for el in line.split('"')[1].split(",")]  # yaw, pitch
        label = np.array([np.double(yaw), np.double(pitch)])
        # name = line[3]
        # gaze2d = line[7]
        # head2d = line[8]
        # lefteye = line[1]
        # righteye = line[2]
        # face = line[0]

        # label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor) # rad

        pitch = label[0] * 180 / np.pi
        yaw = label[1] * 180 / np.pi
        
        img = np.array(img)
        # print(img.shape)
        # img = np.moveaxis(img, -1, 0)
        # img = np.resize(img, [3,95,75])
        
        # print(img.shape)
        # if random.randint(0, 1) == 1:
        #     img, yaw = augmentation(img, yaw)
        # img = Image.fromarray(img)


        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)

        img=torch.from_numpy(img).type(torch.FloatTensor)
        # print(img.shape)

        # Bin values
        angle = 180
        bins = np.array(range(-angle, angle, self.binwidth))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])

        return img, labels, cont_labels

