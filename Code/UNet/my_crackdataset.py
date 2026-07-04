import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(CrackDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "image")) if i.endswith(".jpg")]
        self.img_list = [os.path.join(data_root, "image", i) for i in img_names]
        self.png = [os.path.join(data_root, "label", i.split(".")[0] + ".png")#0开始遍历到结束,我的数据集来说就是png
                       for i in img_names]
        # check files
        for i in self.png:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
#这一段我的数据集没有
    #     self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
    #                      for i in img_names]
    #     # check files
    #     for i in self.roi_mask:
    #         if os.path.exists(i) is False:
    #             raise FileNotFoundError(f"file {i} does not exists.")
    #
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        png = Image.open(self.png[idx]).convert('L')
        png = (255 - np.array(png)) / 255
        # roi_mask = Image.open(self.roi_mask[idx]).convert('L')#没有
        # roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(png, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask) #没有

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

