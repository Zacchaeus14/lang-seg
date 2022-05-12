import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--img', type=str, required=True)
    arg('--gt', type=str, required=True)
    arg('--pred', type=str, required=True)
    return parser.parse_args()


cfg = parse_args()
name = cfg.img.split('/')[-1].split('.')[0]
img = cv2.imread(cfg.img)
gt = cv2.imread(cfg.gt)[:, :, 0]
pred = cv2.imread(cfg.pred)[:, :, 0]

color = np.array([255, 0, 0], dtype='uint8')
masked_img = np.where(gt[..., None], color, img)
out_gt = cv2.addWeighted(img, 0.4, masked_img, 0.6, 0)

color = np.array([0, 0, 255], dtype='uint8')
masked_img = np.where(pred[..., None], color, img)
out_pred = cv2.addWeighted(img, 0.4, masked_img, 0.6, 0)

f, axarr = plt.subplots(1, 3)
for i in range(3):
    axarr[i].axis('off')
    axarr[i].get_xaxis().set_visible(False)
    axarr[i].get_yaxis().set_visible(False)
axarr[0].title.set_text('Image')
axarr[1].title.set_text('Ground Truth')
axarr[2].title.set_text('Prediction')
axarr[0].imshow(img)
axarr[1].imshow(out_gt)
axarr[2].imshow(out_pred)

f.savefig(f'case_analysis/case-{name}.jpg', dpi=400)
