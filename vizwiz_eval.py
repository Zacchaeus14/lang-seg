import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

import wandb


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, help='directory for result masks', required=True)
    parser.add_argument('--label_dir', type=str, help='directory for label masks',
                        default='../datasets/VizWizGrounding2022/binary_masks_png/val/')
    parser.add_argument('--image_dir', type=str, help='directory for images',
                        default='../datasets/VizWizGrounding2022/val/')
    parser.add_argument('--question_path', type=str, help='path to question annotation',
                        default='../datasets/VizWizGrounding2022/val_grounding.json')
    parser.add_argument('--project_name', type=str, help='project name for wandb logging', default='vizwiz-eval')
    parser.add_argument('--entity_name', type=str, help='entity name for wandb logging', default='ych-ycw-capstone')
    return parser.parse_args()


class ViaWizEvaluator:
    def __init__(self, args):
        self.result_dir = args.result_dir
        self.label_dir = args.label_dir
        self.image_dir = args.image_dir
        self.result_fps = glob(os.path.join(self.result_dir, '*.png'))
        self.label_fps = glob(os.path.join(self.label_dir, '*.png'))
        with open(args.question_path, 'r') as f:
            self.annotations = json.load(f)
        self._sanity_check()
        self.names = [x.split('/')[-1].split('.')[0] for x in self.result_fps]
        wandb.init(project=args.project_name, entity=args.entity_name,
                   name=args.result_dir.split('/')[-1].replace('/', ''))
        self.wandb_table = wandb.Table(columns=["name", "question", "answer", "prediction", "iou"])
        self.scores = []

    def _sanity_check(self):
        print("BEGIN SANITY CHECK")
        assert len(self.result_fps) == 1131, f'expecting 1131 results, got {len(self.result_fps)}'
        assert len(self.label_fps) == 1131, f'expecting 1131 results, got {len(self.label_fps)}'
        for fp in tqdm(self.result_fps):
            name = fp.split('/')[-1]
            pred = cv2.imread(fp)[:, :, 0]
            label = cv2.imread(os.path.join(self.label_dir, name))[:, :, 0]
            assert len(np.unique(pred)) <= 2, f'only 0 and 255 should be in mask, but got {np.unique(pred)} in {fp}'
            assert pred.shape == label.shape, f'pred and label should be in the same shape, but got {pred.shape} for pred and {label.shape} for label in {fp}'
        print("DONE SANITY CHECK")

    def _iter_one_pair(self, name):
        image = cv2.imread(os.path.join(self.image_dir, f'{name}.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_gt = cv2.imread(os.path.join(self.label_dir, f'{name}.png'))[:, :, 0]
        mask_pred = cv2.imread(os.path.join(self.result_dir, f'{name}.png'))[:, :, 0]
        iou = ViaWizEvaluator.get_iou(mask_gt, mask_pred)
        question = self.annotations.get(f'{name}.jpg', {}).get('question', '')
        answer = self.annotations.get(f'{name}.jpg', {}).get('most_common_answer', '')
        class_labels = {255: "mask"}
        # image_comp = ViaWizEvaluator.compress(image)
        # print(image.shape, image_comp.shape)
        # print(image_comp)
        # mask_comp = ViaWizEvaluator.compress(mask_pred)
        # print(mask_pred.shape, mask_comp.shape)
        masked_image = wandb.Image(ViaWizEvaluator.compress(image), masks={
            "predictions": {
                "mask_data": ViaWizEvaluator.compress(mask_pred),
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": ViaWizEvaluator.compress(mask_gt),
                "class_labels": class_labels
            }
        }, caption=question)
        self.wandb_table.add_data(name, question, answer, masked_image, iou)
        return iou

    def evaluate(self):
        print('BEGIN EVALUATION')
        for name in tqdm(self.names):
            iou = self._iter_one_pair(name)
            self.scores.append(iou)
        self.log()
        print('mIOU:', np.mean(self.scores))
        print('DONE EVALUATION')

    def log(self):
        wandb.log({"predictions": self.wandb_table})

    @staticmethod
    def compress(image, long_dim=256):
        h, w = image.shape[0], image.shape[1]
        if h > w:
            h1 = long_dim
            w1 = int(long_dim * (w / h))
        else:
            w1 = long_dim
            h1 = int(long_dim * (h / w))
        return cv2.resize(image, (w1, h1))

    @staticmethod
    def get_iou(mask0, mask1):
        i = mask0 & mask1
        u = mask0 | mask1
        i[i == 255] = 1
        u[u == 255] = 1
        return np.sum(i) / np.sum(u)


if __name__ == '__main__':
    args = parse()
    evaluator = ViaWizEvaluator(args)
    evaluator.evaluate()
