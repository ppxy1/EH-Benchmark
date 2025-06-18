import colorsys
import os
import time

import cv2
from PIL import ImageDraw, ImageFont, Image
import random
import numpy as np
import torch
import torch.nn as nn

from torchvision.ops import nms
class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        # -----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        # -----------------------------------------------------------#
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size = 1
            #   batch_size, 3 * (4 + 1 + 80), 20, 20
            #   batch_size, 255, 40, 40
            #   batch_size, 255, 80, 80
            # -----------------------------------------------#
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)

            # -----------------------------------------------#
            #   输入为640x640时
            #   stride_h = stride_w = 32、16、8
            # -----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            # -------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            # -------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]

            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 20, 20, 85
            #   batch_size, 3, 40, 40, 85
            #   batch_size, 3, 80, 80, 85
            # -----------------------------------------------#
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            # -----------------------------------------------#
            #   先验框的中心位置的调整参数
            # -----------------------------------------------#
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            # -----------------------------------------------#
            #   先验框的宽高调整参数
            # -----------------------------------------------#
            w = torch.sigmoid(prediction[..., 2])
            h = torch.sigmoid(prediction[..., 3])
            # -----------------------------------------------#
            #   获得置信度，是否有物体
            # -----------------------------------------------#
            conf = torch.sigmoid(prediction[..., 4])
            # -----------------------------------------------#
            #   种类置信度
            # -----------------------------------------------#
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # ----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角
            #   batch_size,3,20,20
            # ----------------------------------------------------------#
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            # ----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,20,20
            # ----------------------------------------------------------#
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # ----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            # ----------------------------------------------------------#
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

            # ----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            # ----------------------------------------------------------#
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # -----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            # -----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.4):
        # ----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            # ------------------------------------------#
            #   获得预测结果中包含的所有种类
            # ------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                # ------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                # ------------------------------------------#
                detections_class = detections[detections[:, -1] == c]

                # ------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #   筛选出一定区域内，属于同一种类得分最大的框
                # ------------------------------------------#
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]

                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data

                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


class DecodeBoxNP():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBoxNP, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            batch_size = np.shape(input)[0]
            input_height = np.shape(input)[2]
            input_width = np.shape(input)[3]

            # -----------------------------------------------#
            #   输入为640x640时
            #   stride_h = stride_w = 32、16、8
            # -----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            # -------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            # -------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]

            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 20, 20, 85
            #   batch_size, 3, 40, 40, 85
            #   batch_size, 3, 80, 80, 85
            # -----------------------------------------------#
            prediction = np.transpose(
                np.reshape(input, (batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height, input_width)),
                (0, 1, 3, 4, 2))

            # -----------------------------------------------#
            #   先验框的中心位置的调整参数
            # -----------------------------------------------#
            x = self.sigmoid(prediction[..., 0])
            y = self.sigmoid(prediction[..., 1])
            # -----------------------------------------------#
            #   先验框的宽高调整参数
            # -----------------------------------------------#
            w = self.sigmoid(prediction[..., 2])
            h = self.sigmoid(prediction[..., 3])
            # -----------------------------------------------#
            #   获得置信度，是否有物体
            # -----------------------------------------------#
            conf = self.sigmoid(prediction[..., 4])
            # -----------------------------------------------#
            #   种类置信度
            # -----------------------------------------------#
            pred_cls = self.sigmoid(prediction[..., 5:])

            # ----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角
            #   batch_size,3,20,20
            # ----------------------------------------------------------#
            grid_x = np.repeat(np.expand_dims(
                np.repeat(np.expand_dims(np.linspace(0, input_width - 1, input_width), 0), input_height, axis=0), 0),
                               batch_size * len(self.anchors_mask[i]), axis=0)
            grid_x = np.reshape(grid_x, np.shape(x))
            grid_y = np.repeat(np.expand_dims(
                np.repeat(np.expand_dims(np.linspace(0, input_height - 1, input_height), 0), input_width, axis=0).T, 0),
                               batch_size * len(self.anchors_mask[i]), axis=0)
            grid_y = np.reshape(grid_y, np.shape(y))

            # ----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,20,20
            # ----------------------------------------------------------#
            anchor_w = np.repeat(
                np.expand_dims(np.repeat(np.expand_dims(np.array(scaled_anchors)[:, 0], 0), batch_size, axis=0), -1),
                input_height * input_width, axis=-1)
            anchor_h = np.repeat(
                np.expand_dims(np.repeat(np.expand_dims(np.array(scaled_anchors)[:, 1], 0), batch_size, axis=0), -1),
                input_height * input_width, axis=-1)
            anchor_w = np.reshape(anchor_w, np.shape(w))
            anchor_h = np.reshape(anchor_h, np.shape(h))
            # ----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            # ----------------------------------------------------------#
            pred_boxes = np.zeros(np.shape(prediction[..., :4]))
            pred_boxes[..., 0] = x * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h * 2) ** 2 * anchor_h

            # ----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            # ----------------------------------------------------------#
            _scale = np.array([input_width, input_height, input_width, input_height])
            output = np.concatenate([np.reshape(pred_boxes, (batch_size, -1, 4)) / _scale,
                                     np.reshape(conf, (batch_size, -1, 1)),
                                     np.reshape(pred_cls, (batch_size, -1, self.num_classes))], -1)
            outputs.append(output)
        return outputs

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
            计算IOU
        """
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                     np.maximum(inter_rect_y2 - inter_rect_y1, 0)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / np.maximum(b1_area + b2_area - inter_area, 1e-6)

        return iou

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # -----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            # -----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.4):
        # ----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#
        box_corner = np.zeros_like(prediction)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#
            class_conf = np.max(image_pred[:, 5:5 + num_classes], 1, keepdims=True)
            class_pred = np.expand_dims(np.argmax(image_pred[:, 5:5 + num_classes], 1), -1)

            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = np.squeeze((image_pred[:, 4] * class_conf[:, 0] >= conf_thres))

            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not np.shape(image_pred)[0]:
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), 1)

            # ------------------------------------------#
            #   获得预测结果中包含的所有种类
            # ------------------------------------------#
            unique_labels = np.unique(detections[:, -1])

            for c in unique_labels:
                # ------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                # ------------------------------------------#
                detections_class = detections[detections[:, -1] == c]

                # 按照存在物体的置信度排序
                conf_sort_index = np.argsort(detections_class[:, 4] * detections_class[:, 5])[::-1]
                detections_class = detections_class[conf_sort_index]
                # 进行非极大抑制
                max_detections = []
                while np.shape(detections_class)[0]:
                    # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                    max_detections.append(detections_class[0:1])
                    if len(detections_class) == 1:
                        break
                    ious = self.bbox_iou(max_detections[-1], detections_class[1:])
                    detections_class = detections_class[1:][ious < nms_thres]
                # 堆叠
                max_detections = np.concatenate(max_detections, 0)

                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i]
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    # ---------------------------------------------------#
    #   将预测值的每个特征层调成真实值
    # ---------------------------------------------------#
    def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):
        # -----------------------------------------------#
        #   input   batch_size, 3 * (4 + 1 + num_classes), 20, 20
        # -----------------------------------------------#
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        # -----------------------------------------------#
        #   输入为640x640时 input_shape = [640, 640]  input_height = 20, input_width = 20
        #   640 / 20 = 32
        #   stride_h = stride_w = 32
        # -----------------------------------------------#
        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width
        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #   anchor_width, anchor_height / stride_h, stride_w
        # -------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          anchors[anchors_mask[2]]]

        # -----------------------------------------------#
        #   batch_size, 3 * (4 + 1 + num_classes), 20, 20 =>
        #   batch_size, 3, 5 + num_classes, 20, 20  =>
        #   batch_size, 3, 20, 20, 4 + 1 + num_classes
        # -----------------------------------------------#
        prediction = input.view(batch_size, len(anchors_mask[2]),
                                num_classes + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # -----------------------------------------------#
        #   先验框的中心位置的调整参数
        # -----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # -----------------------------------------------#
        #   先验框的宽高调整参数
        # -----------------------------------------------#
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        # -----------------------------------------------#
        #   获得置信度，是否有物体 0 - 1
        # -----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        # -----------------------------------------------#
        #   种类置信度 0 - 1
        # -----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # ----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #   batch_size,3,20,20
        #   range(20)
        #   [
        #       [0, 1, 2, 3 ……, 19],
        #       [0, 1, 2, 3 ……, 19],
        #       …… （20次）
        #       [0, 1, 2, 3 ……, 19]
        #   ] * (batch_size * 3)
        #   [batch_size, 3, 20, 20]
        #
        #   [
        #       [0, 1, 2, 3 ……, 19],
        #       [0, 1, 2, 3 ……, 19],
        #       …… （20次）
        #       [0, 1, 2, 3 ……, 19]
        #   ].T * (batch_size * 3)
        #   [batch_size, 3, 20, 20]
        # ----------------------------------------------------------#
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(y.shape).type(FloatTensor)

        # ----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
        #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
        # ----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # ----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        #   x  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_x
        #   y  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_y
        #   w  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_w
        #   h  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_h
        # ----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
        pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
        pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

        point_h = 5
        point_w = 5

        box_xy = pred_boxes[..., 0:2].cpu().numpy() * 32
        box_wh = pred_boxes[..., 2:4].cpu().numpy() * 32
        grid_x = grid_x.cpu().numpy() * 32
        grid_y = grid_y.cpu().numpy() * 32
        anchor_w = anchor_w.cpu().numpy() * 32
        anchor_h = anchor_h.cpu().numpy() * 32

        fig = plt.figure()
        ax = fig.add_subplot(121)
        from PIL import Image
        img = Image.open("img/street.jpg").resize([640, 640])
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.gca().invert_yaxis()

        anchor_left = grid_x - anchor_w / 2
        anchor_top = grid_y - anchor_h / 2

        rect1 = plt.Rectangle([anchor_left[0, 0, point_h, point_w], anchor_top[0, 0, point_h, point_w]], \
                              anchor_w[0, 0, point_h, point_w], anchor_h[0, 0, point_h, point_w], color="r", fill=False)
        rect2 = plt.Rectangle([anchor_left[0, 1, point_h, point_w], anchor_top[0, 1, point_h, point_w]], \
                              anchor_w[0, 1, point_h, point_w], anchor_h[0, 1, point_h, point_w], color="r", fill=False)
        rect3 = plt.Rectangle([anchor_left[0, 2, point_h, point_w], anchor_top[0, 2, point_h, point_w]], \
                              anchor_w[0, 2, point_h, point_w], anchor_h[0, 2, point_h, point_w], color="r", fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax = fig.add_subplot(122)
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.scatter(box_xy[0, :, point_h, point_w, 0], box_xy[0, :, point_h, point_w, 1], c='r')
        plt.gca().invert_yaxis()

        pre_left = box_xy[..., 0] - box_wh[..., 0] / 2
        pre_top = box_xy[..., 1] - box_wh[..., 1] / 2

        rect1 = plt.Rectangle([pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]], \
                              box_wh[0, 0, point_h, point_w, 0], box_wh[0, 0, point_h, point_w, 1], color="r",
                              fill=False)
        rect2 = plt.Rectangle([pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]], \
                              box_wh[0, 1, point_h, point_w, 0], box_wh[0, 1, point_h, point_w, 1], color="r",
                              fill=False)
        rect3 = plt.Rectangle([pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]], \
                              box_wh[0, 2, point_h, point_w, 0], box_wh[0, 2, point_h, point_w, 1], color="r",
                              fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()
        #


    feat = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()
    anchors = np.array([[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119], [10, 13], [16, 30], [33, 23]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    get_anchors_and_decode(feat, [640, 640], anchors, anchors_mask, 80)

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image, mode='PIL'):
    if mode == 'PIL':
        iw, ih = image.size
        w, h = size

        if letterbox_image:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
    else:
        image = np.array(image)
        if letterbox_image:
            # 获得现在的shape
            shape = np.shape(image)[:2]
            # 获得输出的shape
            if isinstance(size, int):
                size = (size, size)

            # 计算缩放的比例
            r = min(size[0] / shape[0], size[1] / shape[1])

            # 计算缩放后图片的高宽
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = size[1] - new_unpad[0], size[0] - new_unpad[1]

            # 除以2以padding到两边
            dw /= 2
            dh /= 2

            # 对图像进行resize
            if shape[::-1] != new_unpad:  # resize
                image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

            new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                           value=(128, 128, 128))  # add border
        else:
            new_image = cv2.resize(image, (w, h))

    return new_image


# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_classes(classes_path=None):
    class_names = ['ex', 'he', 'se', 'ma']
    return class_names, len(class_names)


# ---------------------------------------------------#
#   获得先验框
# ---------------------------------------------------#
def get_anchors(anchors_path=None):
    anchors = np.array([
        [12, 16], [19, 36], [40, 28],
        [36, 75], [76, 55], [72, 146],
        [142, 110], [192, 243], [459, 401]
    ])
    return anchors, len(anchors)


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   设置种子
# ---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------#
#   设置Dataloader的种子
# ---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def download_weights(phi, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    download_urls = {
        "l": 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
        "x": 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
    }
    url = download_urls[phi]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)

        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)

        x_all = [x_1, x_2]
        # [-1, -3, -5, -6] => [5, 3, 1, 0]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)

        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Transition_Block(nn.Module):
    def __init__(self, c1, c2):
        super(Transition_Block, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)

        self.mp = MP()

    def forward(self, x):
        # 160, 160, 256 => 80, 80, 256 => 80, 80, 128
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)

        # 160, 160, 256 => 160, 160, 128 => 80, 80, 128
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)

        # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
        return torch.cat([x_2, x_1], 1)


class Backbone(nn.Module):
    def __init__(self, transition_channels, block_channels, n, phi, pretrained=False):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        # -----------------------------------------------#
        ids = {
            'l': [-1, -3, -5, -6],
            'x': [-1, -3, -5, -7, -8],
        }[phi]
        # 640, 640, 3 => 640, 640, 32 => 320, 320, 64
        self.stem = nn.Sequential(
            Conv(3, transition_channels, 3, 1),
            Conv(transition_channels, transition_channels * 2, 3, 2),
            Conv(transition_channels * 2, transition_channels * 2, 3, 1),
        )
        # 320, 320, 64 => 160, 160, 128 => 160, 160, 256
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids),
        )
        # 160, 160, 256 => 80, 80, 256 => 80, 80, 512
        self.dark3 = nn.Sequential(
            Transition_Block(transition_channels * 8, transition_channels * 4),
            Multi_Concat_Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids),
        )
        # 80, 80, 512 => 40, 40, 512 => 40, 40, 1024
        self.dark4 = nn.Sequential(
            Transition_Block(transition_channels * 16, transition_channels * 8),
            Multi_Concat_Block(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        )
        # 40, 40, 1024 => 20, 20, 1024 => 20, 20, 1024
        self.dark5 = nn.Sequential(
            Transition_Block(transition_channels * 32, transition_channels * 16),
            Multi_Concat_Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        )

        if pretrained:
            url = {
                "l": 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
                "x": 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # -----------------------------------------------#
        #   dark3的输出为80, 80, 512，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        # -----------------------------------------------#
        #   dark4的输出为40, 40, 1024，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        # -----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        # 输出通道数为c2
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        # -----------------------------------------------#
        #   定义了不同yolov7版本的参数
        # -----------------------------------------------#
        transition_channels = {'l': 32, 'x': 40}[phi]
        block_channels = 32
        panet_channels = {'l': 32, 'x': 64}[phi]
        e = {'l': 2, 'x': 1}[phi]
        n = {'l': 4, 'x': 6}[phi]
        ids = {'l': [-1, -2, -3, -4, -5, -6], 'x': [-1, -3, -5, -7, -8]}[phi]
        conv = {'l': RepConv, 'x': Conv}[phi]
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        # ---------------------------------------------------#
        self.backbone = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)

        # ------------------------加强特征提取网络------------------------#
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 20, 20, 1024 => 20, 20, 512
        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
        # 20, 20, 512 => 20, 20, 256 => 40, 40, 256
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)
        # 40, 40, 1024 => 40, 40, 256
        self.conv_for_feat2 = Conv(transition_channels * 32, transition_channels * 8)
        # 40, 40, 512 => 40, 40, 256
        self.conv3_for_upsample1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4,
                                                      transition_channels * 8, e=e, n=n, ids=ids)

        # 40, 40, 256 => 40, 40, 128 => 80, 80, 128
        self.conv_for_P4 = Conv(transition_channels * 8, transition_channels * 4)
        # 80, 80, 512 => 80, 80, 128
        self.conv_for_feat1 = Conv(transition_channels * 16, transition_channels * 4)
        # 80, 80, 256 => 80, 80, 128
        self.conv3_for_upsample2 = Multi_Concat_Block(transition_channels * 8, panet_channels * 2,
                                                      transition_channels * 4, e=e, n=n, ids=ids)

        # 80, 80, 128 => 40, 40, 256
        self.down_sample1 = Transition_Block(transition_channels * 4, transition_channels * 4)
        # 40, 40, 512 => 40, 40, 256
        self.conv3_for_downsample1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4,
                                                        transition_channels * 8, e=e, n=n, ids=ids)

        # 40, 40, 256 => 20, 20, 512
        self.down_sample2 = Transition_Block(transition_channels * 8, transition_channels * 8)
        # 20, 20, 1024 => 20, 20, 512
        self.conv3_for_downsample2 = Multi_Concat_Block(transition_channels * 32, panet_channels * 8,
                                                        transition_channels * 16, e=e, n=n, ids=ids)
        # ------------------------加强特征提取网络------------------------#

        # 80, 80, 128 => 80, 80, 256
        self.rep_conv_1 = conv(transition_channels * 4, transition_channels * 8, 3, 1)
        # 40, 40, 256 => 40, 40, 512
        self.rep_conv_2 = conv(transition_channels * 8, transition_channels * 16, 3, 1)
        # 20, 20, 512 => 20, 20, 1024
        self.rep_conv_3 = conv(transition_channels * 16, transition_channels * 32, 3, 1)

        # 4 + 1 + num_classes
        # 80, 80, 256 => 80, 80, 3 * 25 (4 + 1 + 20) & 85 (4 + 1 + 80)
        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * 25 & 85
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 512 => 20, 20, 3 * 25 & 85
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)

        # ------------------------加强特征提取网络------------------------#
        # 20, 20, 1024 => 20, 20, 512
        P5 = self.sppcspc(feat3)
        # 20, 20, 512 => 20, 20, 256
        P5_conv = self.conv_for_P5(P5)
        # 20, 20, 256 => 40, 40, 256
        P5_upsample = self.upsample(P5_conv)
        # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512
        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        # 40, 40, 512 => 40, 40, 256
        P4 = self.conv3_for_upsample1(P4)

        # 40, 40, 256 => 40, 40, 128
        P4_conv = self.conv_for_P4(P4)
        # 40, 40, 128 => 80, 80, 128
        P4_upsample = self.upsample(P4_conv)
        # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
        P3 = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        # 80, 80, 256 => 80, 80, 128
        P3 = self.conv3_for_upsample2(P3)

        # 80, 80, 128 => 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 => 40, 40, 256
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 256 => 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 => 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 => 20, 20, 512
        P5 = self.conv3_for_downsample2(P5)
        # ------------------------加强特征提取网络------------------------#
        # P3 80, 80, 128
        # P4 40, 40, 256
        # P5 20, 20, 512

        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)
        # ---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size, 75, 80, 80)
        # ---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        # ---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size, 75, 40, 40)
        # ---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size, 75, 20, 20)
        # ---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)

        return [out0, out1, out2]

'''
训练自己的数据集必看注释！
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : '/raid/baiyang/xiaoyu/yolov7/logs/best_epoch_weights.pth',
        "classes_path"      : 'model_data/voc_classes.txt', 
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #------------------------------------------------------#
        #   所使用到的yolov7的版本，本仓库一共提供两个：
        #   l : 对应yolov7
        #   x : 对应yolov7_x
        #------------------------------------------------------#
        "phi"               : 'x',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.4,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.4,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    # def detect_image(self, image, crop=False, count=False):
    #     # Calculate input image dimensions
    #     image_shape = np.array(np.shape(image)[0:2])
    #     # Convert to RGB
    #     image = cvtColor(image)
    #     # Resize with letterbox
    #     image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
    #     # Add batch dimension
    #     image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    #
    #     with torch.no_grad():
    #         images = torch.from_numpy(image_data)
    #         if self.cuda and torch.cuda.is_available():
    #             images = images.cuda()
    #         # Predict
    #         outputs = self.net(images)
    #         outputs = self.bbox_util.decode_box(outputs)
    #         # Non-max suppression
    #         results = self.bbox_util.non_max_suppression(
    #             torch.cat(outputs, 1), self.num_classes, self.input_shape,
    #             image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou
    #         )
    #
    #     # If no detections, return None
    #     if results[0] is None:
    #         return results
    #
    #     top_label = np.array(results[0][:, 6], dtype='int32')
    #     top_conf = results[0][:, 4] * results[0][:, 5]
    #     top_boxes = results[0][:, :4]
    #
    #     # Font and thickness for drawing
    #     font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #     thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
    #
    #     # Optional counting
    #     if count:
    #         print("top_label:", top_label)
    #         classes_nums = np.zeros([self.num_classes])
    #         for i in range(self.num_classes):
    #             num = np.sum(top_label == i)
    #             if num > 0:
    #                 print(self.class_names[i], " : ", num)
    #             classes_nums[i] = num
    #         print("classes_nums:", classes_nums)
    #
    #     # Optional cropping
    #     if crop:
    #         for i, c in list(enumerate(top_boxes)):
    #             top, left, bottom, right = top_boxes[i]
    #             top = max(0, np.floor(top).astype('int32'))
    #             left = max(0, np.floor(left).astype('int32'))
    #             bottom = min(image.size[1], np.floor(bottom).astype('int32'))
    #             right = min(image.size[0], np.floor(right).astype('int32'))
    #
    #             dir_save_path = "img_crop"
    #             if not os.path.exists(dir_save_path):
    #                 os.makedirs(dir_save_path)
    #             crop_image = image.crop([left, top, right, bottom])
    #             crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
    #             print("save crop_" + str(i) + ".png to " + dir_save_path)
    #
    #     # Draw bounding boxes
    #     for i, c in list(enumerate(top_label)):
    #         predicted_class = self.class_names[int(c)]
    #         box = top_boxes[i]
    #         score = top_conf[i]
    #
    #         top, left, bottom, right = box
    #         top = max(0, np.floor(top).astype('int32'))
    #         left = max(0, np.floor(left).astype('int32'))
    #         bottom = min(image.size[1], np.floor(bottom).astype('int32'))
    #         right = min(image.size[0], np.floor(right).astype('int32'))
    #
    #         label = '{} {:.2f}'.format(predicted_class, score)
    #         draw = ImageDraw.Draw(image)
    #         label_bbox = font.getbbox(label)
    #         label_size = (label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1])
    #         label = label.encode('utf-8')
    #         print(label, top, left, bottom, right)
    #
    #         if top - label_size[1] >= 0:
    #             text_origin = np.array([left, top - label_size[1]])
    #         else:
    #             text_origin = np.array([left, top + 1])
    #
    #         for i in range(thickness):
    #             draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
    #         draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
    #         draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
    #         del draw
    #
    def detect_image(self, image, crop=False, count=False):
        # Calculate input image dimensions
        image_shape = np.array(np.shape(image)[0:2])
        # Convert to RGB
        image = cvtColor(image)
        # Resize with letterbox
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # Add batch dimension
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda and torch.cuda.is_available():
                images = images.cuda()
            # Predict
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # Non-max suppression
            results = self.bbox_util.non_max_suppression(
                torch.cat(outputs, 1), self.num_classes, self.input_shape,
                image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou
            )

        # If no detections, return None
        if results[0] is None:
            return results

        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]

        # Optional counting
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        # Optional cropping
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        return results

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return