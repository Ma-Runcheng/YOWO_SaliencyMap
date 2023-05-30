import argparse

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

from config import build_dataset_config, build_model_config
from dataset.transforms import BaseTransform
from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from models.detector import build_model
from utils.misc import load_weight

from pytorch_grad_cam.pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image
from utils.vis_tools import convert_tensor_to_cv2img
import torchvision.transforms.functional as F
from utils.vis_tools import convert_tensor_to_cv2img, vis_detection
from utils.box_ops import rescale_bboxes

class BaseTransform(object):
    def __init__(self, img_size=224, pixel_mean=[0., 0., 0.], pixel_std=[1., 1., 1.]):
        self.img_size = img_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def to_tensor(self, video_clip):
        return [F.normalize(F.to_tensor(image), self.pixel_mean, self.pixel_std) for image in video_clip]


    def __call__(self, video_clip, normalize=True):
        oh = video_clip[0].height
        ow = video_clip[0].width

        # resize
        video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]

        # to tensor
        video_clip = self.to_tensor(video_clip)

        return video_clip


def parse_args():
    parser = argparse.ArgumentParser(description='YOWO')

    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    # model
    parser.add_argument('--weight', default='./weights/jhmdb21/yowo/yowo_epoch_10.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-v', '--version', default='yowo', type=str, choices=['yowo', 'yowo_nano'],
                        help='build YOWO')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')
    # dataset
    parser.add_argument('-d', '--dataset', default='jhmdb21',
                        help='')
    return parser.parse_args()


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def compute_saliency_maps(X, y, model):
    """
    Input:
    - X: Input images; Tensor of shape (N, 3, H, W), N = 1
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.
    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # Make sure the model is in "test" mode
    model.to(device).eval()
    X.to(device)

    # Make input tensor require gradient
    X.requires_grad_(True)
    loss_dict = model(X, targets=y)
    losses = loss_dict['losses']
    losses.backward()
    gradData = X.grad.data[:, :, 0, :, :]
    saliencies, _ = torch.max(gradData.abs().detach().cpu(), dim=1)
    # x shape:[10, 3, 128, 128] saliencies.shape:[10, 3, 128, 128]

    # We need to normalize each image, because their gradients might vary in scale
    saliency = torch.stack([normalize(item) for item in saliencies])

    return saliency

transform = BaseTransform()
args = parse_args()
# config
d_cfg = build_dataset_config(args)
m_cfg = build_model_config(args)
basetransform = BaseTransform(
    img_size=d_cfg['test_size'],
    pixel_mean=d_cfg['pixel_mean'],
    pixel_std=d_cfg['pixel_std']
)
if args.cuda:
    print('use cuda')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# build model
class_names = d_cfg['label_map']
dataset = UCF_JHMDB_Dataset(
    data_root=d_cfg['data_root'],
    dataset=args.dataset,
    img_size=d_cfg['test_size'],
    transform=basetransform,
    is_train=False,
    len_clip=d_cfg['len_clip'],
    sampling_rate=d_cfg['sampling_rate']
)
num_classes = dataset.num_classes
model = build_model(
    args=args,
    d_cfg=d_cfg,
    m_cfg=m_cfg,
    device=device,
    num_classes=num_classes,
    trainable=False
)
model = load_weight(model=model, path_to_ckpt=args.weight)


def get_video_clip(images):
    img_num = len(images)
    video_clip = [[None for i in range(16)] for j in range(img_num)]
    for img_id in range(1, img_num + 1):
        for i in reversed(range(0, 16)):
            img_id_temp = img_id - i
            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > img_num:
                img_id_temp = img_num
            video_clip[img_id - 1][i - 15] = images[img_id_temp - 1]

    for i in range(0, img_num):
        video_clip[i] = transform(video_clip[i])
        video_clip[i] = torch.stack(video_clip[i], dim=1)
    return video_clip


def get_Attention_Grad_Map(images):
    global model, d_cfg
    video_clip = get_video_clip(images)
    images = []
    device = torch.device("cuda")
    for i in range(len(video_clip)):
        video_clip[i] = torch.unsqueeze(video_clip[i], dim=0)
        video_clip[i] = video_clip[i].to(device)  # [B, 3, T, H, W], B=1
        input_tensor = video_clip[i]
        target_layers = [model.channel_encoder.fuse_convs]
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        grayscale_cam = grayscale_cam[0, :]
        key_frame_tensor = video_clip[i][0, :, 1, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor, d_cfg['pixel_mean'], d_cfg['pixel_std'])
        key_frame = np.float32(key_frame) / 255
        visualization = show_cam_on_image(key_frame, grayscale_cam, use_rgb=True)
        cam_image = visualization
        cam_image = cv2.resize(cam_image, dsize=(320, 240))
        images.append(cam_image)
    return images

def get_Detection_Image(images):
    #video_clip, model, d_cfg = get_video_clip_and_model(images)
    global model, d_cfg
    video_clip = get_video_clip(images)

    orig_size = list(images[0].size)
    images = []
    device = torch.device("cuda")
    model = model.to(device).eval()
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(21)]
    class_names = d_cfg['label_map']
    for i in range(len(video_clip)):
        video_clip[i] = torch.unsqueeze(video_clip[i], dim=0)
        video_clip[i] = video_clip[i].to(device)  # [B, 3, T, H, W], B=1
        batch_scores, batch_labels, batch_bboxes = model(video_clip[i])
        # batch size = 1
        scores = batch_scores[0]
        labels = batch_labels[0]
        bboxes = batch_bboxes[0]

        # rescale
        bboxes = rescale_bboxes(bboxes, orig_size)

        # vis results of key-frame
        key_frame_tensor = video_clip[i][0, :, 1, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor, d_cfg['pixel_mean'], d_cfg['pixel_std'])
        # resize key_frame to orig size
        key_frame = cv2.resize(key_frame, orig_size)
        # visualize detection
        vis_results = vis_detection(
            frame=key_frame,
            scores=scores,
            labels=labels,
            bboxes=bboxes,
            vis_thresh=0.2,
            class_names=class_names,
            class_colors=class_colors
        )
        vis_results = cv2.cvtColor(vis_results, cv2.COLOR_BGR2RGB)
        images.append(vis_results)
    return images

def get_YOLO_Grad_Map(images):
    #video_clip, model, d_cfg = get_video_clip_and_model(images)
    global model, d_cfg
    video_clip = get_video_clip(images)
    images = []
    device = torch.device("cuda")
    for i in range(len(video_clip)):
        video_clip[i] = torch.unsqueeze(video_clip[i], dim=0)
        video_clip[i] = video_clip[i].to(device)  # [B, 3, T, H, W], B=1
        input_tensor = video_clip[i]
        target_layers = [model.backbone_2d.convsets_2]
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        grayscale_cam = grayscale_cam[0, :]
        key_frame_tensor = video_clip[i][0, :, 1, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor, d_cfg['pixel_mean'], d_cfg['pixel_std'])
        key_frame = np.float32(key_frame) / 255
        visualization = show_cam_on_image(key_frame, grayscale_cam, use_rgb=True)
        cam_image = visualization
        cam_image = cv2.resize(cam_image, dsize=(320, 240))
        images.append(cam_image)
    return images

if __name__ == '__main__':
    args = parse_args()
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # transform
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
    )

    dataset = UCF_JHMDB_Dataset(
        data_root=d_cfg['data_root'],
        dataset=args.dataset,
        img_size=d_cfg['test_size'],
        transform=basetransform,
        is_train=False,
        len_clip=d_cfg['len_clip'],
        sampling_rate=d_cfg['sampling_rate']
    )
    class_names = d_cfg['label_map']
    num_classes = dataset.num_classes

    np.random.seed(100)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # build model
    model = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=True
    )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)
    for index in range(0, len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index + 1, len(dataset)))
        frame_id, video_clip, target = dataset[index]
        orig_size = target['orig_size']  # width, height
        # prepare
        video_clip = video_clip.unsqueeze(0).to(device)  # [B, 3, T, H, W], B=1
        input_tensor = video_clip
        # target_layers = [model.backbone_2d.convsets_2]
        # target_layers = [model.backbone_3d.layer4]
        target_layers = [model.channel_encoder.fuse_convs]
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=args.cuda)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        # grayscale_cam = grayscale_cam.squeeze()
        grayscale_cam = grayscale_cam[0, :]
        # grayscale_cam, _ = torch.max(torch.from_numpy(grayscale_cam), dim=2)
        key_frame_tensor = video_clip[0, :, -1, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor, d_cfg['pixel_mean'], d_cfg['pixel_std'])
        key_frame = np.float32(key_frame) / 255
        visualization = show_cam_on_image(key_frame, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cam_image = cv2.resize(cam_image, dsize=(320, 240))
        cv2.imwrite(f'saliencyMap/cam_{index}.jpg', cam_image)

        # saliency = compute_saliency_maps(video_clip, target, model)
        # saliency = saliency.cpu().numpy()
        # saliency = saliency.squeeze()
        # saliency = cv2.resize(saliency, dsize=(340, 240))
        # plt.axis('off')
        # plt.imshow(saliency, cmap=plt.cm.hot)
        # plt.savefig(f'saliencyMap/CGS/cam_{index}.jpg')
