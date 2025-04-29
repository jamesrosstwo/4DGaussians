from io import BytesIO
from pathlib import Path
from typing import List

import decord
import glob
import os
import ffmpeg

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True) # Decord's cuda context doesnt like being reinitialized in forked workers

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
    :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


from pathlib import Path
import ffmpeg


def concatenate_videos(video_list, output_path: Path):
    concat_list_path = output_path.parent / "concat_list.txt"

    with concat_list_path.open('w') as f:
        for video_path in video_list:
            f.write(f"file '{video_path.resolve()}'\n")

    # Intermediate temp file
    temp_concat_path = output_path.parent / "temp_concat.mp4"

    # 1. Concatenate without re-encoding (fast)
    ffmpeg.input(str(concat_list_path), format='concat', safe=0)\
        .output(str(temp_concat_path), c='copy')\
        .run()

    # 2. Re-encode for fast seekability
    ffmpeg.input(str(temp_concat_path))\
        .output(
            str(output_path),
            vcodec='libx264',
            crf=28,               # lower quality, smaller size
            preset='ultrafast',   # fastest encode preset
            g=1,                 # keyframe every 15 frames
            keyint_min=15,        # enforce minimum interval
            sc_threshold=0,       # no scene change keyframes
            pix_fmt='yuv420p'     # standard format
        )\
        .run()

    # Cleanup
    concat_list_path.unlink()
    temp_concat_path.unlink()


class Neural3D_NDC_Dataset(Dataset):
    def __init__(
            self,
            datadir,
            split="train",
            downsample=1.0,
            time_scale=1.0,
            scene_bbox_min=[-1.0, -1.0, -1.0],
            scene_bbox_max=[1.0, 1.0, 1.0],
            eval_index=0,
            width=1352,
            height=1014
    ):
        self.img_wh = (
            int(width / downsample),
            int(height / downsample),
        )  # According to the neural 3D paper, the default resolution is 1024x768
        self.root_dir = datadir
        self._mp4_paths = sorted(Path(self.root_dir).glob("cam*mp4"), key=lambda p: int(p.stem[3:]))

        self.video_path = Path(datadir) / "stitched.mp4"
        if not self.video_path.exists():
            concatenate_videos(self._mp4_paths, self.video_path)

        with open(self.video_path, 'rb') as f:
            video_bytes = f.read()

        video_stream = BytesIO(video_bytes)
        self.reader = decord.VideoReader(video_stream, ctx=decord.gpu())

        self.split = split
        self.downsample = 2 * width / self.img_wh[0]
        self.time_scale = time_scale
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])

        self.eval_index = eval_index
        self.transform = T.ToTensor()

        self.load_meta()
        print(f"meta data loaded, total image:{len(self)}")

    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # Read poses and video file paths.
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        self.near_fars = poses_arr[:, -2:]
        videos = glob.glob(os.path.join(self.root_dir, "cam*.mp4"))
        videos = sorted(videos)
        # breakpoint()
        assert len(videos) == poses_arr.shape[0]

        H, W, focal = poses[0, :, -1]
        focal = focal / self.downsample
        self.focal = [focal, focal]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # poses, _ = center_poses(
        #     poses, self.blender2opencv
        # )  # Re-center poses so that the average is near the center.
        ()
        #
        # near_original = self.near_fars.min scale_factor = near_original * 0.75
        # self.near_fars /= (
        #     scale_factor  # rescale nearest plane so that it is at z = 4/3.
        # )
        # poses[..., 3] /= scale_factor

        # Sample N_views poses for validation - NeRF-like camera trajectory.
        N_views = 300
        self.val_poses = get_spiral(poses, self.near_fars, N_views=N_views)
        # self.val_poses = self.directions
        W, H = self.img_wh
        poses_i_train = []

        for i in range(len(poses)):
            if i != self.eval_index:
                poses_i_train.append(i)
        self.poses = poses[poses_i_train]
        self.poses_all = poses
        self._total_len, self.image_poses, self.image_times, N_cam, N_time = self.load_images_path(videos, self.split)
        self.cam_number = N_cam
        self.time_number = N_time

    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def load_images_path(self, videos, split):
        image_poses = []
        image_times = []
        N_cams = 0
        N_time = 0
        countss = 300
        total_len = 0
        for index, video_path in enumerate(videos):
            if index == self.eval_index:
                if split == "train":
                    continue
            else:
                if split == "test":
                    continue
            N_cams += 1
            count = 0
            video_images_path = video_path.split('.')[0]
            video_frames = cv2.VideoCapture(video_path)

            this_count = 0
            total_len += 300
            for idx in range(300):
                if this_count >= countss: break
                pose = np.array(self.poses_all[index])
                R = pose[:3, :3]
                R = -R
                R[:, 0] = -R[:, 0]
                T = -pose[:3, 3].dot(R)
                image_times.append(idx / countss)
                image_poses.append((R, T))
                # if self.downsample != 1.0:
                #     img = video_frame.resize(self.img_wh, Image.LANCZOS)
                # img.save(os.path.join(image_path,"%04d.png"%count))
                this_count += 1

            #     video_data_save[count] = img.permute(1,2,0)
            #     count += 1
        return total_len, image_poses, image_times, N_cams, 300

    def __len__(self):
        return len(self.image_paths)


    def collate(self, batch_indices: List[int]):
        frames = self.reader.get_batch(batch_indices)
        img = torch.utils.dlpack.from_dlpack(frames.to_dlpack()).permute(0, 3, 1, 2).contiguous() / 255.0
        poses = [self.image_poses[i] for i in batch_indices]
        times = [self.image_times[i] for i in batch_indices]

        return list(torch.unbind(img, dim=0)), poses, times
    

    def load_pose(self, index):
        return self.image_poses[index]
