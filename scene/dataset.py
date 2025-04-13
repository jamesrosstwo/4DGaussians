from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.dataset)


    def collate_fn(self, indices):
        cams = []
        data = self.dataset.collate(indices)

        for index, (image, w2c, time) in zip(indices, zip(*data)):
            R, T = w2c
            FovX = focal2fov(self.dataset.focal[0], image.shape[2])
            FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            mask = None
            cams.append(Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                          image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
                          mask=mask))
        return cams
