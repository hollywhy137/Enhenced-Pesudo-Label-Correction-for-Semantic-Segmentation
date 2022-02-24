import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version
from multiprocessing import Pool
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.robot_dataset import robotDataSet
from collections import OrderedDict
import os
from PIL import Image
from utils.tool import fliplr
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml
import time
from os.path import join
import json

torch.backends.cudnn.benchmark=True

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Oxford_Robot_ICCV19'
DATA_LIST_PATH = './dataset/robot_list/val.txt'
SAVE_PATH = './transition/ROB/results/'

IGNORE_LABEL = 255
NUM_CLASSES = 9
NUM_STEPS = 271 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [
    [70,130,180],
    [220,20,60],
    [119,11,32],
    [0,0,142],
    [220,220,0],
    [250,170,30],
    [70,70,70],
    [244,35,232],
    [128,64,128],
]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--batchsize", type=int, default=12,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()

def save(output_name):
    output, name = output_name
    #output_col = colorize_mask(output)
    output = Image.fromarray(output)

    output.save('%s' % (name))
    #output_col.save('%s_color.png' % (name.split('.jpg')[0]))
    return

def save_heatmap(output_name):
    output, name = output_name
    fig = plt.figure()
    plt.axis('off')
    heatmap = plt.imshow(output, cmap='viridis')
    #fig.colorbar(heatmap)
    fig.savefig('%s_heatmap.png' % (name.split('.jpg')[0]))
    return

def save_scoremap(output_name):
    output, name = output_name
    fig = plt.figure()
    plt.axis('off')
    heatmap = plt.imshow(output, cmap='viridis')
    #fig.colorbar(heatmap)
    fig.savefig('%s_scoremap.png' % (name.split('.jpg')[0]))
    return

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    print(('Num classes', num_classes))
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        if len(label.shape) == 3 and label.shape[2] == 4:
            label = label[:, :, 0]
        if len(label.flatten()) != len(pred.flatten()):
            print(('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                   len(pred.flatten()), gt_imgs[ind],
                                                                                   pred_imgs[ind])))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print(('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist)))))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print((name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2))))
    print(('mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))))
    return mIoUs


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    config_path = os.path.join(args.restore_from,'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    args.model = 'DeepLab'
    gpu0 = args.gpu
    batchsize = args.batchsize

    model_name = os.path.basename(args.restore_from)
    args.save += model_name

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    for step in range(20):
        steps = str((step+1)*5000)
        restore_from = args.restore_from + '/GTA_' + steps + '.pth'
        print('now find iou for model: ', restore_from)
        model = DeeplabMulti(num_classes=args.num_classes, use_se = config['use_se'], train_bn = False, norm_style = config['norm_style'])
        saved_state_dict = torch.load(restore_from)

        try:
            model.load_state_dict(saved_state_dict)
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(saved_state_dict)

        model.eval()
        model.cuda(gpu0)

        th = 960
        tw = 1280
        testloader = data.DataLoader(robotDataSet(args.data_dir, args.data_list, crop_size=(th, tw), resize_size=(tw, th), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                        batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)

        scale = 0.8
        testloader2 = data.DataLoader(robotDataSet(args.data_dir, args.data_list, crop_size=(round(th*scale), round(tw*scale) ), resize_size=( round(tw*scale), round(th*scale)), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                        batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)


        if version.parse(torch.__version__) >= version.parse('0.4.0'):
            interp = nn.Upsample(size=(960, 1280), mode='bilinear', align_corners=True)
        else:
            interp = nn.Upsample(size=(960, 1280), mode='bilinear')

        sm = torch.nn.Softmax(dim = 1)
        log_sm = torch.nn.LogSoftmax(dim = 1)
        kl_distance = nn.KLDivLoss( reduction = 'none')

        for index, img_data in enumerate(zip(testloader, testloader2) ):
            batch, batch2 = img_data
            image, _, _, name = batch
            image2, _, _, name2 = batch2

            inputs = image.cuda()
            inputs2 = image2.cuda()

            with torch.no_grad():
                output1, output2 = model(inputs)
                output_batch = interp(sm(0.5* output1 + output2))

                output1, output2 = model(fliplr(inputs))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))

                del output1, output2, inputs

                output1, output2 = model(inputs2)
                output_batch += interp(sm(0.5* output1 + output2))

                output1, output2 = model(fliplr(inputs2))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))

                del output1, output2, inputs2
                output_batch = output_batch.cpu().data.numpy()


            output_batch = output_batch.transpose(0,2,3,1)
            output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8)
            output_iterator = []


            for i in range(output_batch.shape[0]):
                output_iterator.append(output_batch[i,:,:])
                name_tmp = name[i].split('/')[-1]
                name[i] = '%s/%s' % (args.save, name_tmp)
            with Pool(4) as p:
                p.map(save, zip(output_iterator, name))

            del output_batch

        del model
        save_path = args.save
        compute_mIoU('./data/Oxford_Robot_ICCV19/anno', save_path, 'dataset/robot_list')

    return None

if __name__ == '__main__':
    tt = time.time()
    with torch.no_grad():
        main()
    print('Time used: {} sec'.format(time.time()-tt))


