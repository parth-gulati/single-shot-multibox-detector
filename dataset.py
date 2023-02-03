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
import numpy as np
import os
import math
import cv2
import albumentations as A

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    boxes = []

    for layer in range(len(layers)):
        for i in range(layers[layer]):
            for j in range(layers[layer]):
                mid_x = (i + 0.5) / layers[layer]
                mid_y = (j + 0.5) / layers[layer]
                for k in range(4):
                    if k == 0:
                        w = small_scale[layer]
                        h = small_scale[layer]
                    elif k == 1:
                        w = large_scale[layer]
                        h = large_scale[layer]
                    elif k == 2:
                        w = large_scale[layer] * math.sqrt(2)
                        h = large_scale[layer] / math.sqrt(2)
                    else:
                        w = large_scale[layer] / math.sqrt(2)
                        h = large_scale[layer] * math.sqrt(2)

                    min_x = (mid_x - w / 2) if (mid_x - w / 2) > 0 else 0
                    max_x = (mid_x + w / 2) if (mid_x + w / 2) < 1 else 1
                    min_y = (mid_y - h / 2) if (mid_y - h / 2) > 0 else 0
                    max_y = (mid_y + h / 2) if (mid_y + h / 2) < 1 else 1
                    box = [mid_x, mid_y, w, h, min_x, min_y, max_x, max_y]
                    boxes.append(box)

    new_boxes = np.array(boxes)
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    
    return new_boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max, x_center, y_center):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)

    gx = x_center
    gy = y_center

    ious_greater = ious > threshold

    for i in range(len(ious)):
        if ious[i] > threshold:
            tx = (gx - boxs_default[i, 0]) / boxs_default[i, 2]
            ty = (gy - boxs_default[i, 1]) / boxs_default[i, 3]
            tw = np.log((x_max - x_min)/boxs_default[i, 2])
            th = np.log((y_max - y_min)/boxs_default[i, 3])
            ann_box[i, :] = np.array([tx, ty, tw, th])
            if cat_id == 0:
                ann_confidence[i] = [1, 0, 0, 0]
            elif cat_id == 1:
                ann_confidence[i] = [0, 1, 0, 0]
            elif cat_id == 2:
                ann_confidence[i] = [0, 0, 1, 0]
            else:
                ann_confidence[i] = [0, 0, 0, 1]
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    
    ious_true = np.argmax(ious)
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)

    if True not in ious_greater:
        tx = (gx - boxs_default[ious_true, 0]) / boxs_default[ious_true, 2]
        ty = (gy - boxs_default[ious_true, 1]) / boxs_default[ious_true, 3]
        tw = np.log((x_max - x_min) / boxs_default[ious_true, 2])
        th = np.log((y_max - y_min) / boxs_default[ious_true, 3])
        ann_box[ious_true] = [tx, ty, tw, th]
        if cat_id == 1:
            ann_confidence[ious_true] = [1, 0, 0, 0]
        elif cat_id == 2:
            ann_confidence[ious_true] = [0, 1, 0, 0]
        elif cat_id == 3:
            ann_confidence[ious_true] = [0, 0, 1, 0]
        elif cat_id == 4:
            ann_confidence[ious_true] = [0, 0, 0, 1]

class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, test = False, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        self.transform = A.Compose([
            A.RandomCrop(width=320, height=320),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),],
            bbox_params=A.BboxParams(format='coco'))

        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        self.img_names = sorted(os.listdir(self.imgdir))[: int(0.9 * len(os.listdir(self.imgdir)))]
        self.test_names = sorted(os.listdir(self.imgdir))[int(0.9 * len(os.listdir(self.imgdir))):]
        self.image_size = image_size
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names) if self.train else len(self.test_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"

        img_name = self.imgdir+self.img_names[index] if self.train else self.imgdir+self.test_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt" if self.train else self.anndir+self.test_names[index][:-3]+"txt"

        img = cv2.imread(img_name)
        image_width = img.shape[1]
        image_height = img.shape[0]
        resized_img = cv2.resize(img, (320, 320))
        image = resized_img.transpose(2,0,1)
        annotation = None
        with open(ann_name) as f:
            annotation = [line.rstrip() for line in f]

        for x in annotation:
            anns = x.split(' ')
            classification = int(anns[0])
            x_min = float(anns[1])
            y_min = float(anns[2])
            width = float(anns[3])
            height = float(anns[4])

            x_min /= image_width
            y_min /= image_height
            width /= image_width
            height /= image_height

            x_max = x_min + width
            y_max = y_min + height

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            match(ann_box, ann_confidence, self.boxs_default, self.threshold, classification, x_min, y_min, x_max, y_max, x_center, y_center)
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        # if self.train:
        #     class_labels = []
        #     hidden_values = []
        #     temp_ann_box = []
        #     for i in range(len(ann_confidence)):
        #         for j in range(len(ann_confidence[i])):
        #             if ann_confidence[i][j] == 1 and j!=3:
        #                 class_labels.append(j+1)
        #                 temp_ann_box.append(ann_box[i])
        #             elif ann_confidence[i][j] == 1 and j == 3:
        #                 hidden_values.append(i)
        #
        #     transformed = self.transform(image=image, bboxes=temp_ann_box, class_labels=class_labels)
        #     image = transformed['image']
        #     ann_box = transformed['bboxes']
        #     class_labels = transformed['class_labels']
        #     transformed_conf = np.zeros([self.box_num,self.class_num], np.float32)
        #     for i in range(len(class_labels)):
        #         if class_labels[i] == 1:
        #             transformed_conf[i] = [1, 0, 0, 0]
        #         elif class_labels[i] == 2:
        #             transformed_conf[i] = [0, 1, 0, 0]
        #         elif class_labels[i] == 3:
        #             transformed_conf[i] = [0, 0, 1, 0]
        #         elif class_labels[i] == 4:
        #             transformed_conf[i] = [0, 0, 0, 1]
        #     ann_confidence = transformed_conf
        image = torch.Tensor(image / 255.0)
        return image, ann_box, ann_confidence
