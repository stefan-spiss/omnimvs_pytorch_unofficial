from os.path import join

import cv2
import numpy as np
from ocamcamera import OcamCamera
from torch.utils.data import Dataset
import os


class OmniStereoDataset(Dataset):
    """Omnidirectional Stereo Dataset.
    http://cvlab.hanyang.ac.kr/project/omnistereo/
    """

    def __init__(self, root_dir, filename_txt, transform=None, fov=220):
        self.root_dir = root_dir
        self.transform = transform

        # load filenames
        with open(filename_txt) as f:
            data = f.read()
        self.filenames = data.strip().split('\n')

        # folder name
        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.depth_folder = 'omnidepth_gt_640'

        # load ocam calibration data and generate valid image
        self.ocams = []
        self.valids = []
        for cam in self.cam_list:
            ocam_file = join(root_dir, f'o{cam}.txt')
            self.ocams.append(OcamCamera(ocam_file, fov, show_flag=False))
            self.valids.append(self.ocams[-1].valid_area())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample = {}

        filename = self.filenames[idx]
        # load images
        for i, cam in enumerate(self.cam_list):
            img_path = join(self.root_dir, cam, filename)
            sample[cam] = load_image(img_path, valid=self.valids[i])
        # load inverse depth
        depth_path = join(self.root_dir, self.depth_folder, ('0' + os.path.splitext(filename)[0] + '.tiff'))
        sample['idepth'] = load_invdepth(depth_path)

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_invdepth(filename, min_depth=55):
    '''
    min_depth in [cm]
    '''
    invd_value = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    if not (filename.endswith('.tiff') or filename.endswith('.tif')):
        invdepth = (invd_value / 100.0) / (min_depth * 655) + np.finfo(np.float32).eps
    else:
        invdepth = invd_value
    invdepth *= 100  # unit conversion from cm to m
    return invdepth


def load_image(filename, gray=True, valid=None):
    img = cv2.imread(filename)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not valid is None:
        img[valid == 0] = 0
    return img
