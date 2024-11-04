import numpy as np
import cv2
from os.path import join, isdir, basename, splitext  # make sure to import splitext
from os import listdir, mkdir
from tqdm import tqdm
from filter import GroundNormalFilterIEKF
from visualizer import Visualization
import argparse


def add_image_ids(input_file, output_file):
    """Add image IDs (line numbers) at the start of each line of a pose file."""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile, start=1):
            outfile.write(f"{idx+1} {line}")


def invT(transform):
    """inverse a transform matrix without using np.linalg.inv

    Args:
        transform (ndarray): input transform matrix with shape=(4,4)

    Returns:
        ndarray: output transform matrix with shape=(4,4)
    """
    R_Transposed = transform[:3, :3].T
    result = np.eye(4)
    result[:3, :3] = R_Transposed
    result[:3, 3] = -R_Transposed @ transform[:3, 3]
    return result


def read_kitti_calib(calib_path):
    """Read kitti calibration file and get camera intrinsic matrix

    Args:
        calib_path (string): path to calibration file (xxx/calib.txt)

    Returns:
        ndarray: camera intrinsic matrix with shape=(3,3)
    """
    with open(calib_path, "r") as f:
        lines = f.readlines()
        p2 = lines[2].split()[1:]
        p2 = np.array(p2, dtype=np.float32).reshape(3, 4)
    return p2[:, :3]


def read_kitti_pose(pose_path):
    """Read kitti pose file and get relative transform

    Args:
        pose_path (string): path to pose file

    Returns:
        ndarray: relative transforms with shape=(N,4,4)
    """
    input_pose = np.loadtxt(pose_path)
    assert (input_pose.shape[1] == 13)
    image_ids = input_pose[:, 0].astype(np.int32)
    input_pose = input_pose[:, 1:]
    length = input_pose.shape[0]
    input_pose = input_pose.reshape(-1, 3, 4)
    bottom = np.zeros((length, 1, 4))
    bottom[:, :, -1] = 1
    absolute_transform = np.concatenate((input_pose, bottom), axis=1)
    relative_transforms = []
    for idx in range(length - 1):
        relative_transform = invT(
            absolute_transform[idx + 1]) @ absolute_transform[idx]
        relative_transforms.append(relative_transform)
    return image_ids, relative_transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="00")
    parser.add_argument("--kitti_root", type=str, default="KITTI_odom/sequences")
    parser.add_argument("--pose_root", type=str, default="odometry/orbslam2")
    parser.add_argument("--output_root", type=str, default="results")
    args = parser.parse_args()

    print("parsed!")

    # create output dir
    if not isdir(args.output_root):
        mkdir(args.output_root)
    vis_dir = join(args.output_root, "vis")
    if not isdir(vis_dir):
        mkdir(vis_dir)

    # read poses
    sequence = args.sequence
    pose_file = join(args.pose_root, f"{sequence}.txt")

    # Check if the pose file has 12 or 13 values per line
    with open(pose_file, 'r') as f:
        first_line = f.readline().strip().split()
    
    if len(first_line) == 12:
        print("Pose file has 12 values per line. Adding image IDs...")
        # Create a new filename by appending "_with_ids" to the original file name
        corrected_pose_file = join(args.pose_root, f"{splitext(basename(pose_file))[0]}_with_ids.txt")
        add_image_ids(pose_file, corrected_pose_file)
        pose_file = corrected_pose_file  # Use the corrected file

    # Proceed with reading the pose file
    image_ids, relative_transforms = read_kitti_pose(pose_file)

    # print(relative_transforms)

    ### ADD THE UPDATED IMAGE LIST PROCESSING CODE HERE ###
    # Prepare image list
    image_dir = join(args.kitti_root, sequence, "image_2")
    image_list = sorted(listdir(image_dir))  # Sort the image list to match sequence order

    # Convert image_ids from 1-based to 0-based indexing
    image_ids = image_ids - 1

    # Ensure no index exceeds the length of the image list
    image_ids = image_ids[image_ids < len(image_list)]

    # Filter out images based on image_ids
    image_list = [image_list[i] for i in image_ids]

    if len(image_list) != len(image_ids):
        raise ValueError("Mismatch between image list and image IDs.")

    # read calibration
    calib_path = join(args.kitti_root, sequence, "calib.txt")
    camera_K = read_kitti_calib(calib_path)

    print(f"Sequence: {sequence}, Total images: {len(image_list)}")
    print(f"Kitti root: {args.kitti_root}")
    print(f"Pose root: {args.pose_root}")
    print(f"Output root: {args.output_root}")

    # run
    gnf = GroundNormalFilterIEKF()
    vis = Visualization(K=camera_K, d=None, input_wh=(1241, 376))
    print("Processing images...")
    for idx, image_file in enumerate(tqdm(image_list)):
        image_path = join(image_dir, image_file)
        relative_so3 = relative_transforms[idx][:3, :3]
        compensation_se3 = gnf.update(relative_so3)
        compensation_so3 = compensation_se3[:3, :3]
        image = cv2.imread(image_path)
        combined_image = vis.get_frame(image, compensation_so3)
        output_path = join(vis_dir, f"{idx:06d}.jpg")
        cv2.imwrite(output_path, combined_image)
