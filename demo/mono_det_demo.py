# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
import torch

from mmdet3d.apis import inference_mono_3d_detector, init_model
from mmdet3d.registry import VISUALIZERS
from argparse import ArgumentParser
from mmdet3d.structures.bbox_3d import CameraInstance3DBoxes

import cv2
import numpy as np

import pickle

default_img = 'demo/data/kitti/000008.png' 
default_annote = 'demo/data/kitti/000008.pkl'

#default_cfg = '/home/001891254/CMPE295_mmdetection3d/mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_waymoD5-fov-mono3d_0003_6.py' 
#default_chkpnt = '/home/001891254/CMPE295_mmdetection3d/mmdetection3d/work_dirs/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_waymoD5-fov-mono3d_0003_6/epoch_19.pth' 
default_cfg = '/home/001891254/CMPE295_mmdetection3d/mmdetection3d/configs/smoke/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d_2.py' 

#default_chkpnt = '/home/001891254/CMPE295_mmdetection3d/mmdetection3d/work_dirs/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_kitti-mono3d/epoch_2.pth'
default_chkpnt = '/home/001891254/CMPE295_mmdetection3d/mmdetection3d/work_dirs/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d_2/epoch_4.pth'


def read_ground_truth_from_pkl(file_path):
    """
    Read the ground truth bounding boxes from a .pkl file.

    Parameters:
    - file_path: The path to the .pkl file.

    Returns:
    - gt_bboxes: A list of ground truth 3D bounding boxes.
    """

    with open(file_path, 'rb') as file:
        gt_bboxes = pickle.load(file)

    return gt_bboxes

def project_3d_bbox_to_2d_camera_only(corners_3d, cam2img):
    """
    Project 3D bounding box corners onto the 2D image plane.

    Parameters:
    - corners_3d: Tensor of shape [N, 8, 3] where N is the number of bounding boxes.
    - cam2img: Intrinsic camera matrix.
    - lidar2cam: Transformation matrix from LiDAR to camera coordinates.

    Returns:
    - corners_2d: Projected 2D corners with shape [N, 8, 2].
    """

    # Ensure the corners have homogeneous coordinates
    num_boxes = corners_3d.shape[0]
    ones = torch.ones((num_boxes, 8, 1))
    corners_3d_homogeneous = torch.cat([corners_3d, ones], dim=2)  # [N, 8, 4]

    corners_3d_homogeneous = corners_3d_homogeneous.float()

    #corners_cam_coord = corners_cam_coord.float()
    cam2img = cam2img.float()
    # Project 3D corners in camera coordinates to 2D image plane
    corners_2d_homogeneous = torch.matmul(corners_3d_homogeneous, cam2img.T)  # [N, 8, 4]

    # Perspective division to get 2D coordinates
    corners_2d = corners_2d_homogeneous[:, :, :2] / corners_2d_homogeneous[:, :, 2:3]

    return corners_2d





def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default=default_img, help='image file')
    parser.add_argument('--ann', default=default_annote, help='ann file')
    parser.add_argument('--config', default=default_cfg, help='Config file')
    parser.add_argument('--checkpoint', default=default_chkpnt, help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cam-type',
        type=str,
        default='CAM2',
        help='choose camera type to inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.00, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo/output.png', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data.shape)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # test a single image
    result = inference_mono_3d_detector(model, args.img, args.ann,
                                        args.cam_type)

    #print(result)
    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    data_input = dict(img=img)
    # show the results
    visualizer.add_datasample(
        'result',
        data_input,
        data_sample=result,
        draw_gt=False,
        show=args.show,
        wait_time=-1,
        out_file=args.out_dir,
        pred_score_thr=args.score_thr,
        vis_task='mono_det')

    # Extracting the bounding boxes from the result object
    #bboxes_3d = result["bboxes_3d"].tensor.cpu().numpy()
    bboxes_3d = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    #scores = result["scores_3d"].cpu().numpy()
    scores = result.pred_instances_3d.scores_3d.cpu().numpy()

    image = cv2.imread(args.img)
    
    cam2img = np.array(result.cam2img)
    lidar2cam = np.array(result.lidar2cam)
    cam2img = torch.tensor(cam2img)
    lidar2cam = torch.tensor(lidar2cam)
    #print("cam2img", cam2img)
    #print("lidar2cam", lidar2cam)
    print("scores", scores)

    #cam2img = np.array(result.META_INFORMATION["cam2img"])
    #lidar2cam = np.array(result.META_INFORMATION["lidar2cam"])

    for i, bbox in enumerate(bboxes_3d):
        if scores[i] >= args.score_thr:
            # Extract the 3D bounding box corners
            #bbox_3d_obj = CameraInstance3DBoxes(bbox)
            bbox_3d_obj = CameraInstance3DBoxes(bbox.reshape(1, -1))

            bbox_3d_corners = bbox_3d_obj.corners
            #print("bbox_3d_obj.corners: ", bbox_3d_obj.corners)
            bbox_3d_corners = bbox_3d_corners.squeeze(2)  # or np.squeeze(bbox_3d_corners, axis=2)
            #print("bbox: ", bbox)
            #print("bbox_3d_obj: ", bbox_3d_obj)
            #print("bbox_3d_obj.corners: ", bbox_3d_obj.corners)
            #print("bbox_3d_corners.shape: ", bbox_3d_corners.shape)

            # Project the 3D bounding box corners to 2D image points
            image_points = project_3d_bbox_to_2d_camera_only(bbox_3d_corners, cam2img)
            #print("image_points:",image_points)
            #print("image_points.shape", image_points.shape)


            # Draw the projected 2D bounding box on the image
            for start, end in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
                start_point, end_point = tuple(image_points[0][start].cpu().numpy().astype(int)), tuple(image_points[0][end].cpu().numpy().astype(int))
                image = cv2.line(image, start_point, end_point, color=(0, 255, 0), thickness=2)

    output_path = "output_image.png"
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")
    
    image_gt = cv2.imread(args.img)
    # Read the ground truth bounding boxes from the .pkl file
    gt_bboxes = read_ground_truth_from_pkl(args.ann)  # Assuming the ann file path is the .pkl file containing ground truths
    #print(gt_bboxes)
    # Extract the list of instances from the data
    instances = gt_bboxes['data_list'][0]['instances']
    
    # Extract the 3D bounding boxes from the instances
    gt_bboxes = [instance['bbox_3d'] for instance in instances if 'bbox_3d' in instance]

    # For each ground truth bounding box, project it to the image and draw
    for gt_bbox in gt_bboxes:
        gt_bbox_np = np.array(gt_bbox, dtype=float)  # Convert to numpy array
        gt_bbox_3d_obj = CameraInstance3DBoxes(gt_bbox_np.reshape(1, -1))
        bbox_3d_corners = gt_bbox_3d_obj.corners.squeeze(2)

        # Project the 3D bounding box corners to 2D image points
        image_points = project_3d_bbox_to_2d_camera_only(bbox_3d_corners, cam2img)

        # Draw the projected 2D bounding box on the image, using a different color for distinction
        for start, end in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
            start_point, end_point = tuple(image_points[0][start].cpu().numpy().astype(int)), tuple(image_points[0][end].cpu().numpy().astype(int))
            image_gt = cv2.line(image_gt, start_point, end_point, color=(0, 0, 255), thickness=2)  # Using red for ground truth
    
    output_gt_path = "output_image_gt.png"
    cv2.imwrite(output_gt_path, image_gt)
    print(f"Image saved to {output_gt_path}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
