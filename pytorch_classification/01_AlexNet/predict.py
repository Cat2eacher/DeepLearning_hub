import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AlexNet


def main():
    # ----------------------------------------------------#
    # device
    # ----------------------------------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ----------------------------------------------------#
    #           data_transforms
    # ----------------------------------------------------#
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         ])
    # ----------------------------------------------------#
    #           load image
    # ----------------------------------------------------#
    img_path = "dog.10003.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # ----------------------------------------------------#
    #   创建模型
    # ----------------------------------------------------#
    model = AlexNet(num_classes=2).to(device)

    # ----------------------------------------------------#
    #   导入模型权重
    # ----------------------------------------------------#
    weights_path = "AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()