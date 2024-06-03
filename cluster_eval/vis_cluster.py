import yaml
import os
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from model.cluster_eval import ClusterEval


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def listdir(path):
    name_list = []
    for file in os.listdir(path):
        name_list.append(os.path.join(path, file))
    return name_list


def transform_func(img):
    transform_img = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    img = transform_img(img)
    return img


if __name__ == '__main__':
    # load model
    args = get_config('config/config_eval.yaml')
    net = ClusterEval(args=args).cuda()
    state_dict = torch.load(r'checkpoint/ClusterEval.pt')
    net.load_state_dict(state_dict)

    # load images
    img_path = r'imgs'
    img_path_list = listdir(img_path)

    # vis visualization
    for k, img_path in enumerate(img_path_list):
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_t = transform_func(img).unsqueeze(0).cuda()
        h, w = img_t.shape[2], img_t.shape[3]

        out = net(img_t)
        out = out.float().cpu()
        out = F.interpolate(out, (h, w)).squeeze(0)
        out = out.argmax(dim=0, keepdim=True)
        out = out.permute(1, 2, 0).repeat(1, 1, 3).int().numpy()

        heatmap = None
        heatmap = cv2.normalize(out, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        cv2.imwrite(f'vis_out/{str(k)}_vis_cluster.png', heatmap)
        cv2.imwrite(f'vis_out/{str(k)}_img.png', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        print(k)
