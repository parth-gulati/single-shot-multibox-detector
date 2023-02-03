import math

import numpy as np
import cv2
from dataset import iou

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)]


# use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    # input:
    # windowname      -- the name of the window to display the images
    # pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    # image_          -- the input image to the network
    # boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    _, class_num = pred_confidence.shape
    # class_num = 4
    class_num = class_num - 1
    # class_num = 3 now, because we do not need the last class (background)
    image_ *= 255
    image = np.transpose(image_, (1, 2, 0)).astype(np.uint8)
    image1 = np.zeros(image.shape, np.uint8)
    image2 = np.zeros(image.shape, np.uint8)
    image3 = np.zeros(image.shape, np.uint8)
    image4 = np.zeros(image.shape, np.uint8)
    h, w, _ = image1.shape
    image1[:] = image[:]
    image2[:] = image[:]
    image3[:] = image[:]
    image4[:] = image[:]
    # image1: draw ground truth bounding boxes on image1
    # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    # image3: draw network-predicted bounding boxes on image3
    # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)

    # draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i, j] > 0.5:  # if the network/ground_truth has high confidence on cell[i] with class[j]
                px = boxs_default[i, 0]
                py = boxs_default[i, 1]
                pw = boxs_default[i, 2]
                ph = boxs_default[i, 3]
                dx = ann_box[i, 0]
                dy = ann_box[i, 1]
                dw = ann_box[i, 2]
                dh = ann_box[i, 3]

                gx = pw * dx + px
                gy = ph * dy + py
                gw = pw * np.exp(dw)
                gh = ph * np.exp(dh)

                gt_bounding_start_x = 0 if gx - gw/2 < 0 else gx - gw/2
                gt_bounding_start_y = 0 if gy - gh/2 < 0 else gy - gh/2
                gt_bounding_end_x = 1 if gx + gw/2 > 1 else gx + gw/2
                gt_bounding_end_y = 1 if gy + gh/2 > 1 else gy + gh/2

                gt_default_start_x = 0 if px - pw / 2 < 0 else px - pw / 2
                gt_default_start_y = 0 if py - ph / 2 < 0 else py - ph / 2
                gt_default_end_x = 1 if px + pw / 2 > 1 else px + pw / 2
                gt_default_end_y = 1 if py + ph / 2 > 1 else py + ph / 2

                gt_default_start_x *= image.shape[0]
                gt_default_end_x *= image.shape[0]
                gt_default_start_y *= image.shape[1]
                gt_default_end_y *= image.shape[1]

                gt_bounding_end_y *= image.shape[1]
                gt_bounding_start_y *= image.shape[1]
                gt_bounding_end_x *= image.shape[0]
                gt_bounding_start_x *= image.shape[0]

                gt_bounding_start_x = int(gt_bounding_start_x)
                gt_bounding_end_x = int(gt_bounding_end_x)
                gt_bounding_start_y = int(gt_bounding_start_y)
                gt_bounding_end_y = int(gt_bounding_end_y)

                gt_default_end_x = int(gt_default_end_x)
                gt_default_end_y = int(gt_default_end_y)
                gt_default_start_y = int(gt_default_start_y)
                gt_default_start_x = int(gt_default_start_x)

                cv2.rectangle(image1, (gt_bounding_start_x, gt_bounding_start_y),
                              (gt_bounding_end_x, gt_bounding_end_y), colors[j], 1)
                cv2.rectangle(image2, (gt_default_start_x, gt_default_start_y), (gt_default_end_x, gt_default_end_y), colors[j], 1)
        # TODO:
        # image1: draw ground truth bounding boxes on image1
        # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)

        # you can use cv2.rectangle as follows:
        # start_point = (x1, y1) #top left corner, x1<x2, y1<y2
        # end_point = (x2, y2) #bottom right corner
        # color = colors[j] #use red green blue to represent different classes
        # thickness = 2
        # cv2.rectangle(image?, start_point, end_point, color, thickness)

    # pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i, j] > 0.5:
                px = boxs_default[i, 0]
                py = boxs_default[i, 1]
                pw = boxs_default[i, 2]
                ph = boxs_default[i, 3]
                dx = pred_box[i, 0]
                dy = pred_box[i, 1]
                dw = pred_box[i, 2]
                dh = pred_box[i, 3]

                gx = pw * dx + px
                gy = ph * dy + py
                gw = pw * np.exp(dw)
                gh = ph * np.exp(dh)

                pd_bounding_start_x = 0 if gx - gw / 2 < 0 else gx - gw / 2
                pd_bounding_start_y = 0 if gy - gh / 2 < 0 else gy - gh / 2
                pd_bounding_end_x = 1 if gx + gw / 2 > 1 else gx + gw / 2
                pd_bounding_end_y = 1 if gy + gh / 2 > 1 else gy + gh / 2

                pd_default_start_x = 0 if px - pw / 2 < 0 else px - pw / 2
                pd_default_start_y = 0 if py - ph / 2 < 0 else py - ph / 2
                pd_default_end_x = 1 if px + pw / 2 > 1 else px + pw / 2
                pd_default_end_y = 1 if py + ph / 2 > 1 else py + ph / 2

                pd_default_start_x = int(pd_default_start_x * image.shape[0])
                pd_default_end_x = int(pd_default_end_x * image.shape[0])
                pd_default_start_y = int(pd_default_start_y * image.shape[1])
                pd_default_end_y = int(pd_default_end_y * image.shape[1])

                pd_bounding_end_y = int(pd_bounding_end_y * image.shape[1])
                pd_bounding_start_y = int(pd_bounding_start_y * image.shape[1])
                pd_bounding_end_x = int(pd_bounding_end_x * image.shape[0])
                pd_bounding_start_x = int(pd_bounding_start_x * image.shape[0])

                cv2.rectangle(image4, (pd_bounding_start_x, pd_bounding_start_y),
                              (pd_bounding_end_x, pd_bounding_end_y), colors[j], 1)
                cv2.rectangle(image3, (pd_default_start_x, pd_default_start_y), (pd_default_end_x, pd_default_end_y),
                              colors[j], 1)

        # TODO:
        # image3: draw network-predicted bounding boxes on image3
        # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)

    # combine four images into one
    h, w, _ = image1.shape
    image = np.zeros([h * 2, w * 2, 3], np.uint8)
    image[:h, :w] = image1
    image[:h, w:] = image2
    image[h:, :w] = image3
    image[h:, w:] = image4
    cv2.imshow(windowname + " [[gt_box,gt_dft],[pd_box,pd_dft]]", image)
    cv2.waitKey(1)
    # if you are using a server, you may not be able to display the image.
    # in that case, please save the image using cv2.imwrite and check the saved image for visualization.


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    # TODO: non maximum suppression
    A = np.copy(box_)
    c = np.copy(confidence_)
    cB = []
    B = []
    done = []
    while len(done) < len(A):
        index = np.where(c == c[:, :-1].max())[0][0]
        if index in done:
            return confidence_, A
        class_type = np.where(c == c[:, :-1].max())[1][0]
        if c[index, class_type] < threshold:
             return confidence_, A
        cB.append(c[index])
        c[index] = np.array([0, 0, 0, 0])
        done.append(index)
        B.append(A[index])
        dx, dy, dw, dh = A[index, 0], A[index, 1], A[index, 2], A[index, 3]
        px, py, pw, ph = boxs_default[index, 0], boxs_default[index, 1], boxs_default[index, 2], boxs_default[index, 3]

        gx = pw * dx + px
        gy = ph * dy + py
        gw = pw * np.exp(dw)
        gh = ph * np.exp(dh)

        x_min = 0 if gx - gw/2 < 0 else gx - gw/2
        y_min = 0 if gy - gh/2 < 0 else gy - gh/2
        x_max = 0 if gx + gw / 2 > 1 else gx + gw / 2
        y_max = 0 if gy + gh / 2 > 1 else gy + gh / 2


        for i in range(len(A)):
            if i in done:
                continue
            box_px, box_py, box_pw, box_ph = boxs_default[i, 0], boxs_default[i, 1], boxs_default[i, 2], boxs_default[i, 3]
            box_dx, box_dy, box_dw, box_dh = A[i, 0], A[i, 1], A[i, 2], A[i, 3]
            box_gx = box_pw * box_dx + box_px
            box_gy = box_ph * box_dy + box_py
            box_gw = box_pw * np.exp(box_dw)
            box_gh = box_ph * np.exp(box_dh)

            box_x_min = 0 if box_gx - box_gw / 2 < 0 else box_gx - box_gw / 2
            box_y_min = 0 if box_gy - box_gh / 2 < 0 else box_gy - box_gh / 2
            box_x_max = 0 if box_gx + box_gw / 2 > 1 else box_gx + box_gw / 2
            box_y_max = 0 if box_gy + box_gh / 2 > 1 else box_gy + box_gh / 2

            ious = iou(np.array([[gx, gy, gw, gh, x_min, y_min, x_max, y_max]]), box_x_min, box_y_min, box_x_max, box_y_max)
            if ious[0] >=overlap:
                done.append(i)
                c[i] = np.array([0, 0, 0, 0])
    # input:
    # confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # boxs_default -- default bounding boxes, [num_of_boxes, 8]
    # overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    # threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.

    # output:
    # depends on your implementation.
    # if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    # you can also directly return the final bounding boxes and classes, and write a new visualization function for that.


def generate_mAP(image_no, pred_boxes, pred_confidence, ann_box, ann_confidence, threshold, overlap, boxs_default):
    row = []
    for i in range(len(pred_boxes)):
        class_id = np.argmax(pred_confidence[i])
        if class_id == 3 or pred_confidence[i, class_id] < threshold:
            continue
        TP_or_FP = False

        dx, dy, dw, dh = pred_boxes[i, 0], pred_boxes[i, 1], pred_boxes[i, 2], pred_boxes[i, 3]
        px, py, pw, ph = boxs_default[i, 0], boxs_default[i, 1], boxs_default[i, 2], boxs_default[i, 3]
        gx = pw * dx + px
        gy = ph * dy + py
        gw = pw * np.exp(dw)
        gh = ph * np.exp(dh)

        x_min = 0 if gx - gw / 2 < 0 else gx - gw / 2
        y_min = 0 if gy - gh / 2 < 0 else gy - gh / 2
        x_max = 0 if gx + gw / 2 > 1 else gx + gw / 2
        y_max = 0 if gy + gh / 2 > 1 else gy + gh / 2

        for j in range(len(ann_box)):
            if np.argmax(ann_confidence[j]) == class_id:

                box_px, box_py, box_pw, box_ph = boxs_default[j, 0], boxs_default[j, 1], boxs_default[j, 2], boxs_default[j, 3]
                box_dx, box_dy, box_dw, box_dh = ann_box[j, 0], ann_box[j, 1], ann_box[j, 2], ann_box[j, 3]
                box_gx = box_pw * box_dx + box_px
                box_gy = box_ph * box_dy + box_py
                box_gw = box_pw * np.exp(box_dw)
                box_gh = box_ph * np.exp(box_dh)

                box_x_min = 0 if box_gx - box_gw / 2 < 0 else box_gx - box_gw / 2
                box_y_min = 0 if box_gy - box_gh / 2 < 0 else box_gy - box_gh / 2
                box_x_max = 0 if box_gx + box_gw / 2 > 1 else box_gx + box_gw / 2
                box_y_max = 0 if box_gy + box_gh / 2 > 1 else box_gy + box_gh / 2

                ious = iou(np.array([[gx, gy, gw, gh, x_min, y_min, x_max, y_max]]), box_x_min, box_y_min, box_x_max, box_y_max)
                if ious[0] >= overlap:
                    TP_or_FP = True
                    break
            row.append([image_no, pred_boxes[i, class_id], TP_or_FP])
        print(row)
    return row










