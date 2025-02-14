import argparse
import numpy as np

import torch
import imageio
from datetime import datetime
from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule

from PIL import Image, ImageOps
from encoding.models.sseg import BaseNet
import torchvision.transforms as transforms
import json
from tqdm import tqdm
from pathlib import Path
import cv2
import os

def load_model():
    class Options:
        def __init__(self):
            parser = argparse.ArgumentParser(description="PyTorch Segmentation")
            # model and dataset
            parser.add_argument(
                "--model", type=str, default="encnet", help="model name (default: encnet)"
            )
            parser.add_argument(
                "--backbone",
                type=str,
                default="clip_vitl16_384",
                help="backbone name (default: resnet50)",
            )
            parser.add_argument(
                "--dataset",
                type=str,
                default="vizwiz",
                help="dataset name (default: pascal12)",
            )
            parser.add_argument(
                "--workers", type=int, default=16, metavar="N", help="dataloader threads"
            )
            parser.add_argument(
                "--base-size", type=int, default=520, help="base image size"
            )
            parser.add_argument(
                "--crop-size", type=int, default=480, help="crop image size"
            )
            parser.add_argument(
                "--train-split",
                type=str,
                default="train",
                help="dataset train split (default: train)",
            )
            parser.add_argument(
                "--aux", action="store_true", default=False, help="Auxilary Loss"
            )
            parser.add_argument(
                "--se-loss",
                action="store_true",
                default=False,
                help="Semantic Encoding Loss SE-loss",
            )
            parser.add_argument(
                "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
            )
            parser.add_argument(
                "--batch-size",
                type=int,
                default=16,
                metavar="N",
                help="input batch size for \
                                training (default: auto)",
            )
            parser.add_argument(
                "--test-batch-size",
                type=int,
                default=16,
                metavar="N",
                help="input batch size for \
                                testing (default: same as batch size)",
            )
            # cuda, seed and logging
            parser.add_argument(
                "--no-cuda",
                action="store_true",
                default=False,
                help="disables CUDA training",
            )
            parser.add_argument(
                "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
            )
            # checking point
            parser.add_argument(
                "--weights", type=str, default='', help="checkpoint to test"
            )
            # evaluation option
            parser.add_argument(
                "--eval", action="store_true", default=False, help="evaluating mIoU"
            )
            parser.add_argument(
                "--export",
                type=str,
                default=None,
                help="put the path to resuming file if needed",
            )
            parser.add_argument(
                "--acc-bn",
                action="store_true",
                default=False,
                help="Re-accumulate BN statistics",
            )
            parser.add_argument(
                "--test-val",
                action="store_true",
                default=False,
                help="generate masks on val set",
            )
            parser.add_argument(
                "--no-val",
                action="store_true",
                default=False,
                help="skip validation during training",
            )

            parser.add_argument(
                "--module",
                default='lseg',
                help="select model definition",
            )

            # test option
            parser.add_argument(
                "--data-path", type=str, default='../datasets/', help="path to test image folder"
            )

            parser.add_argument(
                "--no-scaleinv",
                dest="scale_inv",
                default=True,
                action="store_false",
                help="turn off scaleinv layers",
            )

            parser.add_argument(
                "--widehead", default=False, action="store_true", help="wider output head"
            )

            parser.add_argument(
                "--widehead_hr",
                default=False,
                action="store_true",
                help="wider output head",
            )
            parser.add_argument(
                "--ignore_index",
                type=int,
                default=-1,
                help="numeric value of ignore label in gt",
            )

            parser.add_argument(
                "--label_src",
                type=str,
                default="default",
                help="how to get the labels",
            )

            parser.add_argument(
                "--arch_option",
                type=int,
                default=0,
                help="which kind of architecture to be used",
            )

            parser.add_argument(
                "--block_depth",
                type=int,
                default=0,
                help="how many blocks should be used",
            )

            parser.add_argument(
                "--activation",
                choices=['lrelu', 'tanh'],
                default="lrelu",
                help="use which activation to activate the block",
            )
            parser.add_argument(
                '--split',
                default='val',
                choices=['val', 'test']
            )

            self.parser = parser

        def parse(self):
            args = self.parser.parse_args()
            args.cuda = not args.no_cuda and torch.cuda.is_available()
            print(args)
            return args

    args = Options().parse()
    print('args:', args)
    torch.manual_seed(args.seed)
    args.test_batch_size = 1
    alpha = 0.5

    args.scale_inv = False
    args.widehead = True
    args.dataset = 'vizwiz'
    # args.backbone = 'clip_vitl16_384'
    # args.weights = 'checkpoints/result-epoch=10-val_acc_epoch=0.83.ckpt'
    args.ignore_index = 255
    print('dataset:', args.dataset)
    print('weight path:', args.weights)
    module = LSegModule.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path=args.data_path,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
        dropout=0.0,
        scale_inv=args.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=args.widehead,
        widehead_hr=args.widehead_hr,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )

    input_transform = module.val_transform

    # dataloader
    loader_kwargs = (
        {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
    )

    # model
    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module

    model = model.eval()
    model = model.cpu()
    scales = (
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        if args.dataset == "citys"
        else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )

    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]
    if torch.cuda.is_available():
        evaluator = LSeg_MultiEvalModule(
            model, scales=scales, flip=True
        ).cuda()
    else:
        evaluator = LSeg_MultiEvalModule(
            model, scales=scales, flip=True
        ).cpu()
    evaluator.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize([512, 512]),
        ]
    )

    return evaluator, transform, args


"""
# LSeg Demo
"""
lseg_model, lseg_transform, args = load_model()
with open(f'../datasets/VizWizGrounding2022/{args.split}_grounding.json', 'r') as f:
    test_json = json.load(f)
todaystring = datetime.now().strftime("%Y%m%d-%H%M%S")
directory = f"results/{args.split}/{todaystring}/"
print('results will be save to:', directory)
Path(directory).mkdir(parents=True, exist_ok=True)
for fn, data in tqdm(test_json.items()):
    fp = f'../datasets/VizWizGrounding2022/{args.split}/{fn}'
    question = data['question']
    labels = ['other', question]
    image = Image.open(fp)
    image = ImageOps.exif_transpose(image)
    width, height = image.size
    pimage = lseg_transform(np.array(image)).unsqueeze(0)
    if torch.cuda.is_available():
        pimage = pimage.cuda()
    # print('labels:', labels)

    with torch.no_grad():
        outputs = lseg_model.forward(pimage, labels)
        print('output shape:', outputs.cpu().numpy().shape)  # [bs=1, 2, h, w]
        predicts = [
            torch.max(output, 0)[1].cpu().numpy()
            for output in outputs
        ]
        # predicts = torch.max(outputs, 1).cpu().numpy()
        # print('predict shape:', np.array(predicts).shape)

    image = pimage[0].permute(1, 2, 0)
    image = image * 0.5 + 0.5
    image = image.cpu()
    image = Image.fromarray(np.uint8(255 * image)).convert("RGBA")

    pred = np.array(predicts[0])
    # print('pred shape:', np.array(pred).shape)
    # print('pred:', pred)
    # print('pred unique:', np.unique(pred, return_counts=True))
    pred = pred.astype(np.uint8)
    pred[pred==1] = 255
    resized = cv2.resize(pred,(width,height), interpolation = cv2.INTER_NEAREST)

    # post-processing
    if np.all(resized==0):
        resized += 255
    # print('resized unique:', np.unique(resized))
    imageio.imwrite(os.path.join(directory, fn.replace('jpg', 'png')), resized)
