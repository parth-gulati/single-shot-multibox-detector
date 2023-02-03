import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    pred_conf = torch.reshape(pred_confidence, (-1, 4))
    pred_boxes = torch.reshape(pred_box, (-1, 4))
    ann_conf = torch.reshape(ann_confidence, (-1, 4))
    ann_boxes = torch.reshape(ann_box, (-1, 4))

    isobj = torch.where(ann_conf[:, -1] == 0)
    noobj = torch.where(ann_conf[:, -1] == 1)

    loss_conf_isobj = F.cross_entropy(pred_conf[isobj], ann_conf[isobj])
    loss_conf_noobj = 3 * F.cross_entropy(pred_conf[noobj], ann_conf[noobj])
    loss_box = F.smooth_l1_loss(pred_boxes, ann_boxes)
    loss = loss_box + loss_conf_isobj + loss_conf_noobj

    return loss
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.

class Convolution(nn.Module):
    def __init__(self, cin, cout, kernel_size, step_size, padding):
        super(Convolution, self).__init__()
        self.cin = cin
        self.cout = cout
        self.kernel_size = kernel_size
        self.step_size = step_size

        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=step_size, padding=padding),
            nn.BatchNorm2d(cout),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class SSD(nn.Module):
    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background

        layers = [
            nn.Conv2d(3,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]

        cin = 64

        while cin < 512:
            for i in range(2):
                layers.append(Convolution(cin, cin, 3, 1, 1))
            layers.append(Convolution(cin, 2*cin, 3, 2, 1))
            cin *= 2
        for i in range(2):
            layers.append(Convolution(512, 512, 3, 1, 1))
        layers.append(Convolution(512, 256, 3, 2, 1))

        self.convolution = nn.Sequential(*layers)

        self.conv_left_1 = nn.Sequential(
            Convolution(256, 256, 1, 1, 0),
            Convolution(256, 256, 3, 2, 1)
        )

        self.conv_left_2 = nn.Sequential(
            Convolution(256, 256, 1, 1, 0),
            Convolution(256, 256, 3, 1, 0)
        )

        self.bottom_left_red_1 = nn.Conv2d(256, 16, 1, 1, 0)
        self.bottom_left_blue_1 = nn.Conv2d(256, 16, 1, 1, 0)

        self.right_red_1 = nn.Conv2d(256, 16, 3, 1, 1)
        self.right_blue_1 = nn.Conv2d(256, 16, 3, 1, 1)

        self.right_red_2 = nn.Conv2d(256, 16, 3, 1, 1)
        self.right_blue_2 = nn.Conv2d(256, 16, 3, 1, 1)

        self.right_red_3 = nn.Conv2d(256, 16, 3, 1, 1)
        self.right_blue_3 = nn.Conv2d(256, 16, 3, 1, 1)

        self.softmax = nn.Softmax(2)

    def forward(self, x):
        x = x / 255.0
        temp = self.convolution(x)
        left_res_1 = self.conv_left_1(temp)
        left_res_2 = self.conv_left_2(left_res_1)
        left_res_3 = self.conv_left_2(left_res_2)

        left_bottom_red_res_1 = self.bottom_left_red_1(left_res_3)
        left_bottom_red_res_1 = torch.reshape(left_bottom_red_res_1, (-1, 16, 1))

        left_bottom_blue_res_1 = self.bottom_left_blue_1(left_res_3)
        left_bottom_blue_res_1 = torch.reshape(left_bottom_blue_res_1, (-1, 16, 1))

        right_red_res_1 = self.right_red_1(temp)
        right_red_res_1 = torch.reshape(right_red_res_1, (-1, 16, 100))

        right_blue_res_1 = self.right_blue_1(temp)
        right_blue_res_1 = torch.reshape(right_blue_res_1, (-1, 16, 100))

        right_red_res_2 = self.right_red_2(left_res_1)
        right_red_res_2 = torch.reshape(right_red_res_2, (-1, 16, 25))

        right_blue_res_2 = self.right_blue_2(left_res_1)
        right_blue_res_2 = torch.reshape(right_blue_res_2, (-1, 16, 25))

        right_red_res_3 = self.right_red_3(left_res_2)
        right_red_res_3 = torch.reshape(right_red_res_3, (-1, 16, 9))

        right_blue_res_3 = self.right_blue_3(left_res_2)
        right_blue_res_3 = torch.reshape(right_blue_res_3, (-1, 16, 9))

        red_final = torch.cat((left_bottom_red_res_1, right_red_res_1, right_red_res_2, right_red_res_3), 2)
        blue_final = torch.cat((left_bottom_blue_res_1, right_blue_res_1, right_blue_res_2, right_blue_res_3), 2)

        red_final = torch.permute(red_final, (0, 2, 1))
        bboxes = torch.reshape(red_final, (-1, 540, 4))

        blue_final = torch.permute(blue_final, (0, 2, 1))
        blue_final = torch.reshape(blue_final, (-1, 540, 4))
        confidence = blue_final

        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        #x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence,bboxes










