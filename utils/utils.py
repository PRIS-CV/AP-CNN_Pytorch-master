import yaml
import torch
import numpy as np
import cv2

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# plot multi-level ROIs
def plot_roi(inputs, roi_list, unorm, vis, mode='train'):
    with torch.no_grad():
        color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        for i in range(inputs.size(0)):
            img = unorm(inputs.data[i].cpu()).numpy().copy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img = np.transpose(img, [1, 2, 0])
            r, g, b = cv2.split(img)
            img = cv2.merge([b, g, r])
            for j, roi in enumerate(roi_list):
                roi = roi[roi[:, 0] == i]
                if len(roi.size()) == 1:
                    b = roi.data.cpu().numpy()
                    cv2.rectangle(img, (b[1], b[2]), (b[3], b[4]), color[j % len(color)], 2)
                else:
                    for k in range(roi.size(0)):
                        b = roi[k].data.cpu().numpy()
                        cv2.rectangle(img, (b[1], b[2]), (b[3], b[4]), color[j % len(color)], 2)
            img = np.transpose(img, [2, 0, 1])
            vis.img('%s_img_%d' % (mode, i), img)

# plot attention masks
def plot_mask_cat(inputs, mask_cat, unorm, vis, mode='train'):
    with torch.no_grad():
        for i in range(inputs.size(0)):
            img = unorm(inputs.data[i].cpu()).numpy().copy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img = np.transpose(img, [1, 2, 0])
            r, g, b = cv2.split(img)
            img = cv2.merge([b, g, r])
            img = np.transpose(img, [2, 0, 1])
            vis.img('%s_img_%d' % (mode, i), img)
            for j in range(mask_cat.size(1)):
                mask = mask_cat[i, j, :, :].data.cpu().numpy()
                img_mask = (255.0 * (mask - np.min(mask)) / (np.max(mask) - np.min(mask))).astype(np.uint8)
                # img_mask = (255.0 * mask).astype(np.uint8)
                img_mask = cv2.resize(img_mask, dsize=(448, 448))
                vis.img('%s_img_%d_mask%d' % (mode, i, j), img_mask)