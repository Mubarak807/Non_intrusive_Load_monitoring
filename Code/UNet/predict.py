import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "F:\郭文龙\模型/UNet\训练/12\save_weights 1\model_177.pth"
    input_folder = "F:/郭文龙/识别结果\陈村2025-11-25"  # 输入文件夹路径
    output_folder = "F:\郭文龙\识别结果\陈村2025-11-25"  # 输出文件夹路径
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(input_folder), f"input folder {input_folder} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)
    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    model.eval()  # 进入验证模式

    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpeg', '.jpg', '.bmp', '.tiff'))]

    total_time = 0

    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)

        # load image
        original_img = Image.open(img_path).convert('RGB')

        # 从pil image到tensor并归一化
        img = data_transform(original_img)
        # 扩展batch维度
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()

            inference_time = t_end - t_start
            total_time += inference_time
            print(f"{image_file} inference time: {inference_time:.3f}s")

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            # ===================== 核心修改：正确映射裂缝和背景 =====================
            # 模型输出：0 = 背景，1 = 裂缝
            # 目标映射：背景 → 白色（255），裂缝 → 黑色（0）
            prediction_mask = np.zeros_like(prediction)  # 初始化全黑
            prediction_mask[prediction == 0] = 255  # 背景类（0）→ 白色（255）
            prediction_mask[prediction == 1] = 0    # 裂缝类（1）→ 黑色（0）
            # ====================================================================

            # 生成输出文件名（保持原文件名，扩展名为.png）
            output_filename = os.path.splitext(image_file)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)

            mask = Image.fromarray(prediction_mask)  # 使用修改后的mask
            mask.save(output_path)
            print(f"Saved result to {output_path}")

    print(f"Processed {len(image_files)} images in total")
    print(f"Average inference time: {total_time / len(image_files):.3f}s per image")


if __name__ == '__main__':
    main()