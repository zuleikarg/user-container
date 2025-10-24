from data import COCODetection, get_label_map, MEANS, COLORS
from yolact.yolact_s import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import threading

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})
package_path = get_package_share_directory('yolact')
trained_model = os.path.join(package_path, 'weights', 'yolact_plus_resnet50_54_800000.pth')
top_k = 1
cuda = True
fast_nms = True
cross_class_nms = True
display_masks = True
display_bboxes = True
display_text = True
display_scores = True
display = False
shuffle = False
ap_data_file = 'results/ap_data.pkl'
resume = False
max_images = -1
output_coco_json = False
bbox_det_file = 'results/bbox_detections.json'
mask_det_file = 'results/mask_detections.json'
config = None
output_web_json = False
web_det_path = 'web/dets/'
no_bar = False
display_lincomb = False
benchmark = False
no_sort = False
seed = None
mask_proto_debug = False
crop = False
video_multiframe = 1
score_threshold = 0.5
dataset = None
detect = False
display_fps= False
emulate_playback = False

#########################################################################################################
### This code has been adapted from the original YOLACT repository: https://github.com/dbolya/yolact
### Copyright (c) 2019 Daniel Bolya, Colin R. Raffel, and others. MIT License.

### Modifications for ROS2 integration and OpticalFlow adaptation by Zuleika M. Redondo García, 2025.

# The steps were taken by following the original DIO-SLAM paper: https://doi.org/10.3390/s24185929
#   - For this goal, the frame information is taken from a ROS2 topic, which would correspond to the RGB camera feed of the robot.
#   - Then, thanks to the segmenetaion of YOLACT, the non-rigid and rigid objects are segmented in the scene, separately.
#   - The segmentation masks are published into three different ROS2 topics: nonrigid_segmentation, rigid_segmentation and camera_corr.
#       -> Nonrigid objects: humans, animals, plants, etc.
#       -> Rigid objects: furniture, vehicles, tools, etc.
#       -> Camera_corr: original RGB image from the camera to allow synchronization with other implementations in the pipieline, such as Optical Flow.

#########################################################################################################


class Segmentation(Node):

    def __init__(self):
        # Initialize node
        super().__init__('seg_node')

        # Create publishers
        self.nonrigid_seg_ = self.create_publisher(Image, 'nonrigid_segmentation', 10)
        self.rigid_seg_ = self.create_publisher(Image, 'rigid_segmentation', 10)
        self.cam_corr_ = self.create_publisher(Image, 'camera_corr', 10)

        # Thread lock for camera frame
        self.lock = threading.Lock()
        self.camera_frame_ = np.empty(0)

        # CvBridge for image conversion
        self.bridge = CvBridge()

        # Get trained model and set configuration
        self.trained_model = trained_model

        self.model_path = SavePath.from_str(self.trained_model)
        self.config = self.model_path.model_name + '_config'
        set_cfg(self.config)

        # Avoid overusing GPU memory
        torch.cuda.empty_cache()

        with torch.no_grad():
            if not os.path.exists('results'):
                os.makedirs('results')

            if torch.cuda.is_available():
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')
            

            self.dataset = None 

            # Load the model
            print('Loading model...', end='')
            self.net = Yolact()
            self.net.load_weights(self.trained_model)
            self.net.eval()
            print(' Done.')

            # Move the model to GPU if available
            if torch.cuda.is_available():
                self.net = self.net.cuda()

        self.processing_thread = threading.Thread(target=self.main_pipeline)
        self.processing_thread.daemon = True

        # Create camera subscription
        self.camera = self.create_subscription(
            Image,
            '/robot/camera/rgb/image_raw',
            self.camera_callback,
            10)
        
        # Start processing thread to avoid blocking of camera callback
        self.processing_thread.start()

    # CAMERA CALLBACK
    def camera_callback(self,msg):
        with self.lock:
            if (msg.height != 0):
                self.camera_frame_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
  
    # MAIN PIPELINE:
    #   - Main loop that processes incoming camera frames with YOLACT model
    #   - Segmentation of non-rigid and rigid objects from the camera frame
    def main_pipeline(self):    
        try:
            while rclpy.ok():
                with self.lock:
                    if self.camera_frame_.size > 0:
                        with torch.no_grad():
                            evaluate(self.net, None, self.nonrigid_seg_, self.rigid_seg_,self.cam_corr_, self.camera_frame_)
                            self.get_logger().info("Iteration finished.")
                    else:
                        time.sleep(0.01)  # avoid busy-waiting

        # Once the code is stoped, stop and join the thread properly
        except (KeyboardInterrupt, ExternalShutdownException):
            self.processing_thread.stop()
            self.processing_thread.join()
            exit()
            pass


# PREPARATION FOR DISPLAY
# Modifications:
#  - The function now returns three images:
#       -> img_numpy: original image with masks, bounding boxes and labels drawn
#       -> img_nonrigid_numpy: binary image with non-rigid objects segmented in white
#       -> img_rigid_numpy: binary image with rigid objects segmented in white
#  - The segmentation masks for non-rigid and rigid objects are created based on COCO dataset class indices.
#  - The images are processed to create binary masks and clean them up using morphological operations.
def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    # Thanks to postprocessing the score_threshold is applied to masks, boxes, and classes separately
    # This way, it can keep some masks that passes the threshold regardless if their boxes do not pass it
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = display_lincomb,
                                        crop_masks        = crop,
                                        score_threshold   = score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    # The detections are reduced based on score threshold
    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color
        
    # Get all the masks for non-rigid elements as white segments
    def get_nonrigid(j, on_gpu=None):
            nonrigid_indices = {0,14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24} # Indices on COCO dataet for non-rigid objects
            if classes[j] in nonrigid_indices:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)

            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.

            return color
    
    # Get all the masks for rigid elements as white segments
    def get_rigid(j, on_gpu=None):
            nonrigid_indices = {0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24} # Indices on COCO dataset for rigid objects
            if classes[j] not in nonrigid_indices:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)

            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.

            return color

    # Copy the img_gpu stored for the segmented images as black frames
    img_nonrigid = img_gpu.clone().detach() *0
    img_rigid = img_gpu.clone().detach() *0

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        nonrigid_segments = torch.cat([get_nonrigid(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        rigid_segments = torch.cat([get_rigid(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)

        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
        masks_nonrigid = masks.repeat(1, 1, 1, 3) * nonrigid_segments
        masks_rigid = masks.repeat(1, 1, 1, 3) * rigid_segments

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # Composite the image with the masks one by one
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        masks_nonrigid_summand = masks_nonrigid[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_nonrigid_cumul = masks_nonrigid[1:] * inv_alph_cumul
            masks_nonrigid_summand += masks_nonrigid_cumul.sum(dim=0)

        masks_rigid_summand = masks_rigid[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_rigid_cumul = masks_rigid[1:] * inv_alph_cumul
            masks_rigid_summand += masks_rigid_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        img_nonrigid = img_nonrigid * inv_alph_masks.prod(dim=0) + masks_nonrigid_summand
        img_rigid = img_rigid * inv_alph_masks.prod(dim=0) + masks_rigid_summand
    
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    img_nonrigid_numpy = (img_nonrigid * 255).byte().cpu().numpy()
    img_rigid_numpy = (img_rigid * 255).byte().cpu().numpy()

    if display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Transform nonrigid and rigid images into BINARY IMAGES
    # Convert to grayscale
    img_nonrigid_numpy = cv2.cvtColor(img_nonrigid_numpy, cv2.COLOR_BGR2GRAY)
    img_rigid_numpy = cv2.cvtColor(img_rigid_numpy, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, img_nonrigid_numpy = cv2.threshold(img_nonrigid_numpy, 127, 255, cv2.THRESH_BINARY)
    _, img_rigid_numpy = cv2.threshold(img_rigid_numpy, 127, 255, cv2.THRESH_BINARY)   
    
    # Apply morphological operations to clean up the masks
    kernel = np.ones((7,7),np.uint8) 

    img_nonrigid_numpy = cv2.morphologyEx(img_nonrigid_numpy, cv2.MORPH_CLOSE, kernel)
    img_rigid_numpy = cv2.morphologyEx(img_rigid_numpy, cv2.MORPH_CLOSE, kernel)

    if display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        return img_numpy, img_nonrigid_numpy, img_rigid_numpy

    if display_text or display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
    
    return img_numpy, img_nonrigid_numpy, img_rigid_numpy

def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=crop, score_threshold=score_threshold)

    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:top_k] for x in t]
        if isinstance(scores, list):
            box_scores = scores[0].cpu().numpy()
            mask_scores = scores[1].cpu().numpy()
        else:
            scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        boxes = boxes.cpu().numpy()
        masks = masks.cpu().numpy()
    
    with timer.env('Sync'):
        # Just in case
        torch.cuda.synchronize()

def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]

def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


class Detections:

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x)*10)/10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })
    
    def dump(self):
        dump_arguments = [
            (self.bbox_data, bbox_det_file),
            (self.mask_data, mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)
    
    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                        'use_yolo_regressors', 'use_prediction_matching',
                        'train_masks']

        output = {
            'info' : {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)
        

        

def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()

def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()

def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections:Detections=None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h*w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes  , gt_boxes   = split(gt_boxes)
                crowd_masks  , gt_masks   = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=crop, score_threshold=score_threshold)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores
        masks = masks.view(-1, h*w).cuda()
        boxes = boxes.cuda()


    if output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i,:],   box_scores[i])
                    detections.add_mask(image_id, classes[i], masks[i,:,:], mask_scores[i])
            return
    
    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt   = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item(),
                     lambda i,j: crowd_bbox_iou_cache[i,j].item(),
                     lambda i: box_scores[i], box_indices),
            ('mask', lambda i,j: mask_iou_cache[i, j].item(),
                     lambda i,j: crowd_mask_iou_cache[i,j].item(),
                     lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
        
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)
                
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue
                    
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                            
                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue
                                
                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x =  ((x >> 16) ^ x) & 0xFFFFFFFF
    return x

# EVAL IMAGE FUNCTION
# Modifications:
#  - After getting the predictions send to prep_display and get the segmented images for non-rigid and rigid objects.
#  - If save_path is None, prepare and publish the segmented images to ROS2 topics
def evalimage(net:Yolact, path:str,camera_frame, nonrigid_pub, rigid_pub, cam_corr,save_path:str=None):
    # Send the image to the GPU and apply the transformations and YOLACT model
    frame = torch.from_numpy(camera_frame).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    # Use the predictions to prepare the images for display
    img_numpy, img_nonrigid, img_rigid = prep_display(preds, frame, None, None, undo_transform=False)
    
    if save_path is None:
        # Prepare and publish images to ROS2 topics
        img_numpy = img_numpy[:, :, (2, 1, 0)]
        img_nonrigid = img_nonrigid
        img_rigid = img_rigid
        
        nr_s = Image()
        r_s = Image()
        cam = Image()

        bridge = CvBridge()
        nr_s = bridge.cv2_to_imgmsg(img_nonrigid, encoding='mono8')
        r_s = bridge.cv2_to_imgmsg(img_rigid, encoding='mono8')
        cam = bridge.cv2_to_imgmsg(camera_frame, encoding='rgb8')

        nonrigid_pub.publish(nr_s)
        rigid_pub.publish(r_s)
        cam_corr.publish(cam)

        cv2.imshow('YOLACT Segmentation', img_numpy)
        cv2.imshow('Non-Rigid Segmentation', img_nonrigid)
        cv2.imshow('Rigid Segmentation', img_rigid)
        cv2.waitKey(1)
    

    # if save_path is None:
    #     plt.imshow(img_numpy)
    #     plt.title(path)
    #     plt.show()
    # else:
    #     cv2.imwrite(save_path, img_numpy)

# def evalimages(net:Yolact, input_folder:str, output_folder:str):
#     if not os.path.exists(output_folder):
#         os.mkdir(output_folder)

#     #print()
#     for p in Path(input_folder).glob('*'): 
#         path = str(p)
#         name = os.path.basename(path)
#         name = '.'.join(name.split('.')[:-1]) + '.png'
#         out_path = os.path.join(output_folder, name)

#         evalimage(net, path, out_path)
#         print(path + ' -> ' + out_path)
#     print('Done.')

# from multiprocessing.pool import ThreadPool
# from queue import Queue


# Main eval function
# Modifications:
#   - Adapted to only process a single image from a camera frame instead of giving various options 
#       such as video, real-time video or stream of several images.
#   - Removed all code related to calculating mAP and other metrics.
#   - Handling of ROS2 publishers. 
def evaluate(net:Yolact, dataset, nonrigid_pub, rigid_pub,cam_corr, camera_frame, train_mode=False):
    net.detect.use_fast_nms = fast_nms
    net.detect.use_cross_class_nms = cross_class_nms
    cfg.mask_proto_debug = mask_proto_debug

    evalimage(net,None, camera_frame, nonrigid_pub, rigid_pub,cam_corr)


    frame_times = MovingAverage()
    dataset_size = -1
    progress_bar = ProgressBar(30, dataset_size)

    #print()

    if not display and not benchmark:
        # For each class and iou, stores tuples (score, isPositive)
        # Index ap_data[type][iouIdx][classIdx]
        ap_data = {
            'box' : [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
        }
        detections = Detections()
    else:
        timer.disable('Load Data')

    dataset_indices = list(range(dataset_size))


def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))
    
    print_maps(all_maps)
    
    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


def main(args=None):
    # Start the fusion node
    try:
        rclpy.init(args=args)
        segmentation_node = Segmentation()

        rclpy.spin(segmentation_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if segmentation_node is not None:
            segmentation_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
            cv2.destroyAllWindows()
            
            segmentation_node.get_logger().info("Segmentation node has been shut down.")


if __name__ == '__main__':
    main()