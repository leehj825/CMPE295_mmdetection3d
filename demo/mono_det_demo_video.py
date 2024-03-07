# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
import mmengine
import torch

from mmdet3d.apis import inference_mono_3d_detector, init_model
from mmdet3d.registry import VISUALIZERS
from argparse import ArgumentParser
from mmdet3d.structures.bbox_3d import CameraInstance3DBoxes

import cv2
import numpy as np
import pickle
import os
import imageio
from PIL import Image, ImageDraw

#default_folder = '/scratch/cmpe249-fa23/kitti_od/training/image_2' 
#default_annote = '/scratch/cmpe249-fa23/kitti_od/kitti_infos_train.pkl'

default_folder = '/scratch/cmpe249-fa23/waymo_data/waymo_single/kitti_format/training/image_0' 
default_annote = '/scratch/cmpe249-fa23/waymo_data/waymo_single/kitti_format/waymo_infos_train.pkl'

#default_cfg = '/home/001891254/295a_project/CMPE295_mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_waymoD5-fov-mono3d_0.py' 
default_cfg = '/home/001891254/295a_project/CMPE295_mmdetection3d/configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_all.py'
#default_cfg = '/home/001891254/295a_project/CMPE295_mmdetection3d/configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_1.py'

#default_chkpnt = '/home/001891254/295a_project/CMPE295_mmdetection3d/work_dirs/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_waymoD5-fov-mono3d_0/20240116_174835/epoch_59.pth'
default_chkpnt = '/home/001891254/295a_project/CMPE295_mmdetection3d/work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_all/epoch_49.pth'
#default_chkpnt = '/home/001891254/295a_project/CMPE295_mmdetection3d/work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d_1/epoch_50.pth'

#output_file_name = 'output_fcos3d_waymo_0_epoch_59.gif'
output_file_name = 'output_pgd_waymo_all_epoch_49.gif'
#output_file_name = 'output_fcos3d_waymokitti_epoch_50.gif'


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
    parser.add_argument('--folder', default=default_folder, help='folder with images')
    parser.add_argument('--ann', default=default_annote, help='ann file')
    parser.add_argument('--config', default=default_cfg, help='Config file')
    parser.add_argument('--checkpoint', default=default_chkpnt, help='Checkpoint file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--cam-type',
        type=str,
        default='CAM2',
        help='choose camera type to inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.2, help='bbox score threshold')
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

    #data_list = mmengine.load(args.ann)['data_list']
    #print("data_list", data_list)
    # Load the annotations for all images
    #all_annotations = read_ground_truth_from_pkl(args.ann)
    #print("loading annotation file, ", args.ann)
    #annotations_list = mmengine.load(args.ann)['data_list'] # Extract the list of annotations
    
    # Load the annotations and verify
    #print("Loading annotation file:", args.ann)
    try:
        annotations_list = mmengine.load(args.ann)['data_list']
        if not annotations_list:
            raise ValueError("Annotation list is empty.")
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return
    
    # Debug: Print the number of annotations loaded
    print(f"Number of annotations loaded: {len(annotations_list)}")

    # Loop through annotations list and print each annotation's data
    #for idx, annotation in enumerate(annotations_list):
    #    print(f"Annotation {idx}: {annotation}")
        
    # Get list of images in the folder
    image_paths = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Debug: Print the number of images found
    print(f"Number of images found: {len(image_paths)}")

    # Sort the images (if necessary) to maintain order
    image_paths.sort()

    # Check if there are images
    if not image_paths:
        print("No images found in the folder.")
        return

    # Initialize video writer
    sample_img = cv2.imread(image_paths[0])
    height, width, layers = sample_img.shape
    #video_name = 'output_video.mp4'
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))
    
    #gif_name = 'output_video.gif'
    gif_frames = []
    
    for index, data_info in enumerate(annotations_list):
        cam_type = args.cam_type

        if index > 50:
            continue
        # Check if the camera type is valid for the current data_info
        if cam_type in data_info['images']:
            # Construct the full path by concatenating the base directory and the file name
            img_file_name = data_info['images'][cam_type]['img_path']
            full_img_path = os.path.join(default_folder, img_file_name)
            
            print(f"Processing image: {full_img_path}")
            #print("data_info['images'][cam_type]", data_info['images'][cam_type])

            # Proceed with your image processing using 'full_img_path'
            try:
                result = inference_mono_3d_detector(model, full_img_path, [data_info], cam_type)

                #result = inference_mono_3d_detector(model, img_path, current_annotation, args.cam_type)
            except Exception as e:
                print(f"Error during inference on image {full_img_path}: {e}")
                continue
            
            #result = inference_mono_3d_detector(model, img_path, current_annotation,
            #                                args.cam_type)
            
            # Read and process the image (similar to the original script)
            #frame = cv2.imread(full_img_path)
            frame = Image.open(full_img_path)
            draw = ImageDraw.Draw(frame)
            
            # Extracting the bounding boxes from the result object
            #bboxes_3d = result["bboxes_3d"].tensor.cpu().numpy()
            bboxes_3d = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            #scores = result["scores_3d"].cpu().numpy()
            scores = result.pred_instances_3d.scores_3d.cpu().numpy()
            
            
            cam2img = np.array(result.cam2img)
            lidar2cam = np.array(result.lidar2cam)
            cam2img = torch.tensor(cam2img)
            lidar2cam = torch.tensor(lidar2cam)
            print("cam2img", cam2img)
            #print("lidar2cam", lidar2cam)
            print("scores", scores)
            #print("bboxes_3d", np.array2string(bboxes_3d, formatter={'float_kind':lambda x: "%.3f" % x}))
            
            formatted_bboxes_3d = [[f"{coordinate:.3f}" for coordinate in bbox] for bbox in bboxes_3d]
            #print("scoformatted_bboxes_3dres", formatted_bboxes_3d)
            #for bbox in formatted_bboxes_3d:
                #print("formatted_bbox_3d", bbox)
            

            #cam2img = np.array(result.META_INFORMATION["cam2img"])
            #lidar2cam = np.array(result.META_INFORMATION["lidar2cam"])

            for i, bbox in enumerate(bboxes_3d):
                if scores[i] >= args.score_thr:
                    print("formatted_bbox_3d", bbox)
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
                    #image_points = project_3d_bbox_to_2d_camera_only(bbox_3d_corners, cam2img)
                    
                    # Error handling for projection
                    try:
                        image_points = project_3d_bbox_to_2d_camera_only(bbox_3d_corners, cam2img)
                    except Exception as e:
                        print(f"Error in projecting 3D bbox to 2D for image {img_path}: {e}")
                        continue
                    #print("image_points:",image_points)
                    #print("image_points.shape", image_points.shape)


                    # Draw the projected 2D bounding box on the image
                    for start, end in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
                        start_point, end_point = tuple(image_points[0][start].cpu().numpy().astype(int)), tuple(image_points[0][end].cpu().numpy().astype(int))
                        #frame = cv2.line(frame, start_point, end_point, color=(0, 255, 0), thickness=2)
                        draw.line([start_point, end_point], fill="green", width=2)
            # Write the frame to the video
            gif_frames.append(frame)
            
            #output_frame_path = f'{output_file_name}_{index}.gif'
            #cv2.imwrite(output_frame_path, frame)
            #print(f"Image saved to {output_frame_path}")
        #video.write(frame)

    # Release the video writer
    #video.release()
    #imageio.mimsave(args.out_dir, gif_frames, format='GIF', fps=5)   
    gif_frames[0].save(output_file_name, save_all=True, append_images=gif_frames[1:], duration=200, loop=0)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
