import os
import torch

from src import UNet
from train_utils import evaluate
from my_crackdataset import CrackDataset
import transforms as T


class SegmentationPresetEval:
    def __init__(self, base_size=565, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = CrackDataset(args.data_path, train=False,
                                  transforms=SegmentationPresetEval(mean=mean, std=std)
                                  )

    num_workers = 0
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             shuffle=False,
                                             collate_fn=val_dataset.collate_fn)

    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    model.to(device)

    confmat,dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
    print(confmat)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 validation")

    parser.add_argument("--data-path", default="D:/7/sjj/小论文最后测试", help="VOCdevkit root")
    parser.add_argument("--weights", default="C:/Users/Administrator/Desktop/UNet/save_weights 2/model_195.pth")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
