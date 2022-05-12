import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json


def parse_args():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--img', type=str, required=True)
    arg('--gt', type=str, required=True)
    arg('--pred', type=str, required=True)
    arg('--anno', type=str, default='/Users/yuchenwang/git/datasets/VizWizGrounding2022/val_grounding.json')
    return parser.parse_args()

def get_iou(mask0, mask1):
    i = mask0 & mask1
    u = mask0 | mask1
    i[i == 255] = 1
    u[u == 255] = 1
    return np.sum(i) / np.sum(u)

cfg = parse_args()
name = cfg.img.split('/')[-1].split('.')[0]
print(name)
img = cv2.imread(cfg.img)
gt = cv2.imread(cfg.gt)[:, :, 0]
pred = cv2.imread(cfg.pred)[:, :, 0]
if np.all(pred > 0):
    pred[:,:] = 0
iou = get_iou(gt, pred)
with open(cfg.anno, 'r') as f:
    anno = json.load(f)
question = anno[f'{name}.jpg']['question']
answer = anno[f'{name}.jpg']['most_common_answer']

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

f.tight_layout()
# # f.suptitle('This is a somewhat long figure title', fontsize=16)
# w, h = f.get_size_inches()*f.dpi # get fig size in pixels
# w = int(w)
# h = int(h)
# print(f"figure size: w = {w}, h = {h}")
f.text(0.5, 0.19, f"Q: {question}", ha='center', va='center')
f.text(0.5, 0.145, f"A: {answer}", ha='center', va='center')
f.text(0.5, 0.10, "IoU: {:.2f}".format(iou), ha='center', va='center', color='g')

f.savefig(f'case_analysis/case-{name}.jpg', dpi=400)
