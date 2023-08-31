import argparse
import json

import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from dataloader import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from models import OmniMVS
from models import SphericalSweeping
from utils import InvDepthConverter

parser = argparse.ArgumentParser(description='Training for OmniMVS',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('root_dir', metavar='DATA_DIR', help='path to dataset')
parser.add_argument('model', metavar='MODEL', help='path to trained model')
parser.add_argument('-i', '--input-list', default='./dataloader/sunny_val.txt',
                    type=str, help='Text file includes filenames for input')
parser.add_argument('--fov', type=float, default=220, help='field of view of the camera in degree')
if False:
    # Paper setting
    parser.add_argument('--input_width', type=int, default=800, metavar='N', help='input image width')
    parser.add_argument('--input_height', type=int, default=768, metavar='N', help='input image height')
else:
    # Light weight
    parser.add_argument('--input_width', type=int, default=500, metavar='N', help='input image width')
    parser.add_argument('--input_height', type=int, default=480, metavar='N', help='input image height')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N', help='number of data loading workers')

running = True

def main():
    args = parser.parse_args()
    print('Arguments:')
    print(json.dumps(vars(args), indent=1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    if device.type != 'cpu':
        cudnn.benchmark = True
    print("device:", device)

    ###############################
    # Load checkpoint

    checkpoint = torch.load(args.model)
    ndisp = checkpoint['ndisp']
    min_depth = checkpoint['min_depth']
    output_width = checkpoint['output_width']
    output_height = checkpoint['output_height']

    # Setup model
    sweep = SphericalSweeping(args.root_dir, h=output_height, w=output_width, fov=args.fov)
    model = OmniMVS(sweep, ndisp, min_depth, h=output_height, w=output_width)
    invd_min = model.inv_depths[0]
    invd_max = model.inv_depths[-1]
    converter = InvDepthConverter(ndisp, invd_min, invd_max)

    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    # model = nn.DataParallel(model)

    # Setup dataloader
    image_size = (args.input_width, args.input_height)
    depth_size = (output_width, output_height)
    train_transform = transforms.Compose([Resize(image_size, depth_size), ToTensor(), Normalize()])
    input_set = OmniStereoDataset(args.root_dir, args.input_list, transform=train_transform, fov=args.fov)
    print(f'{len(input_set)} samples for inference.')
    input_loader = DataLoader(input_set, batch_size=1, shuffle=False, num_workers=args.workers)

    ToPIL = lambda x: transforms.ToPILImage()(x.cpu())


    fig, ax = plt.subplots(3, 1, figsize=(12,12), subplot_kw=({"xticks":(), "yticks":()}))

    def on_close(event):
        global running
        running = False

    fig.canvas.mpl_connect('close_event', on_close)
    plt.ion()
    plt.show()

    ###############################
    # Start inference
    ###############################
    print("Start inference")

    for imgs_in in input_loader:
        model.eval()
        # with torch.no_grad:
        for key in imgs_in.keys():
            imgs_in[key] = imgs_in[key].to(device)
        pred = model(imgs_in)
        gt_idepth = imgs_in['idepth']
        gt_invd_idx = converter.invdepth_to_index(gt_idepth, round_value=False)

        idx = 0
        imgs = []
        for cam in model.cam_list:
            imgs.append(0.5*imgs_in[cam][idx]+0.5)
        img_grid = ToPIL(make_grid(imgs, padding=5, pad_value=1))

        pred_vis = ToPIL(pred[idx]/ndisp)
        gt_vis = ToPIL(gt_invd_idx[idx]/ndisp)

        cmap='viridis'
        # fig.clf()
        ax[0].set_title('fisheye images')
        ax[0].imshow(img_grid)
        ax[1].set_title('prediction')
        ax[1].imshow(pred_vis, cmap=cmap)
        ax[2].set_title('groudtruth')
        ax[2].imshow(gt_vis, cmap=cmap)
        plt.tight_layout()
        plt.draw()
        
        while running:
            plt.waitforbuttonpress(1)

        if not plt.fignum_exists(fig.number):
            break

if __name__ == '__main__':
    main()

