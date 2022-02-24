import numpy as np
import os
from model.deeplab_multi import DeeplabMulti
import torch
import yaml
from torch.utils import data
from dataset.robot_dataset import robotDataSet
from utils.tool import fliplr
from packaging import version
from PIL import Image
import argparse
import torch.nn as nn


torch.backends.cudnn.benchmark=True
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def gen_pn_plabel(args, restore_from):

    config_path = os.path.join(os.path.dirname(restore_from), 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    args.model = config['model']
    gpu0 = args.gpu
    batchsize = 4

    model = DeeplabMulti(num_classes=args.num_classes, use_se=config['use_se'], train_bn=False,
                         norm_style=config['norm_style'])
    saved_state_dict = torch.load(restore_from)

    try:
        model.load_state_dict(saved_state_dict)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(gpu0)

    # to enable mc-dropout, set the dropout layer as train mode
    if args.mc_dropout:
        repeat = 5
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    else:
        repeat = 1

    testloader = data.DataLoader(
        robotDataSet(args.data_dir, args.data_list, crop_size=(960, 1280), resize_size=(1280, 960), mean=IMG_MEAN,
                          scale=False, mirror=False, set=args.set),
        batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)

    scale = 1.25
    testloader2 = data.DataLoader(
        robotDataSet(args.data_dir, args.data_list, crop_size=(round(960 * scale), round(1080 * scale)),
                          resize_size=(round(1280 * scale), round(960 * scale)), mean=IMG_MEAN, scale=False,
                          mirror=False, set=args.set),
        batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(960, 1280), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(1960, 1280), mode='bilinear')

    sm = torch.nn.Softmax(dim=1)
    print('start to generate pseudo labels using' + str(restore_from))
    neg_percent = 0
    posi_ignore = 0
    posi_ignore_min = 0
    pic_count = 0

    for index, img_data in enumerate(zip(testloader, testloader2)):

        batch, batch2 = img_data
        image, _, _, name = batch
        image2, _, _, name2 = batch2

        inputs = image.cuda()
        inputs2 = image2.cuda()

        final_output = []
        #final_output_negative = []
        for i in range(inputs.shape[0]):
            final_output.append([])
            #final_output_negative.append([])


        for n in range(repeat):
            with torch.no_grad():
                output1, output2 = model(inputs)
                output12 = 0.5 * output1 + output2
                output_batch = interp(sm(output12))
                #output_batch_negative = interp(sm(output12/args.temp_scale))
                del output1, output2, output12

                output1, output2 = model(fliplr(inputs))
                output1, output2 = fliplr(output1), fliplr(output2)
                output12 = 0.5 * output1 + output2
                output_batch += interp(sm(output12))
                #output_batch_negative += interp(sm(output12 / args.temp_scale))
                del output1, output2, output12

                output1, output2 = model(inputs2)
                output12 = 0.5 * output1 + output2
                output_batch += interp(sm(output12))
                #output_batch_negative += interp(sm(output12 / args.temp_scale))
                del output1, output2, output12

                output1, output2 = model(fliplr(inputs2))
                output1, output2 = fliplr(output1), fliplr(output2)
                output12 = 0.5 * output1 + output2
                output_batch += interp(sm(output12))
                #output_batch_negative += interp(sm(output12 / args.temp_scale))
                del output1, output2, output12

                output_batch = output_batch.cpu().data.numpy()
                output_batch = output_batch.transpose(0, 2, 3, 1) #(batchsize, height, width, class)

                #output_batch_negative = output_batch_negative.cpu().data.numpy()
                #output_batch_negative = output_batch_negative.transpose(0, 2, 3, 1)

                for j in range(output_batch.shape[0]):
                    final_output[j].append(output_batch[j])
                    #final_output_negative[j].append(output_batch_negative[j])
                del output_batch #, output_batch_negative

        for k in range(len(final_output)):
            pic_count += 1
            final_output_k = final_output[k]
            name_k = name[k]
            output = np.stack(final_output_k)
            mean = np.mean(output, axis=0)
            minimum = np.min(output, axis = 0)
            #std = np.std(output, axis=0)
            #std.round(4)
            label = np.argmax(mean, axis=2)
            label_min = np.argmax(minimum, axis = 2)

            #std_out = np.zeros(label.shape)
            #label_std = label.copy()

            label_prob = np.max(mean, axis = 2)
            label_prob_min = np.max(minimum, axis = 2)
            label_prob.round(4)
            label_prob_min.round(4)

            for h in range(label.shape[0]):
                for w in range(label.shape[1]):
                    index = label[h][w]
                    if mean[h][w][index] < 4*0.8: # or std[h][w][index] > 4*0.05: #if probability is too small or the variance is too large
                        label[h][w] = 255
                        #label_std[h][w] = 255
                        posi_ignore += 1
                    if minimum[h][w][index] < 4*0.8: # or std[h][w][index] > 4*0.05: #if probability is too small or the variance is too large
                        label_min[h][w] = 255
                        posi_ignore_min += 1
                    #if std[h][w][index] > 4*0.05:
                        #label_std[h][w] = 255
                    #std_out[h][w] = std[h][w][index]

            #final_output_negative_k = final_output_negative[k]
            #output_negative = np.stack(final_output_negative_k)
            #mean_negative = 4 - np.mean(output_negative, axis=0)
            #std_negative = np.std(output_negative, axis=0)

            #out_neg_1 = (mean_negative >= 4*0.95).astype(int)
            #out_neg_2 = (std_negative < 4*0.005).astype(int)
            #out_neg = np.minimum(out_neg_1, out_neg_2)
            #neg_percent += np.sum(out_neg)/(out_neg.shape[0]*out_neg.shape[1]*out_neg.shape[2])
            #out_neg.astype(bool)

            label = np.asarray(label, dtype=np.uint8)
            #label_std = np.asarray(label_std, dtype=np.uint8)
            label_min = np.asarray(label_min, dtype=np.uint8)
            #label_negative = np.asarray(label_negative, dtype=np.uint8)
            label = Image.fromarray(label)
            #label_std = Image.fromarray(label_std)
            label_min = Image.fromarray(label_min)
            #label_negative = Image.fromarray(label_negative)

            name_tmp = name_k.split('/')[-1]
            dir_name = name_k.split('/')[-2]
            save_path = args.save_path + dir_name
            #save_path_negative = './transition/GTA/pseudo/negative_1_2/' + dir_name
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            #if not os.path.isdir(save_path_negative):
                #os.mkdir(save_path_negative)
            label.save('%s/%s' % (save_path, name_tmp))
            #label_std.save('%s/%s_2.png' % (save_path, name_tmp[:-4]))
            label_min.save('%s/%s_2.png' % (save_path, name_tmp[:-4]))
            np.savez_compressed('%s/%s' % (save_path, name_tmp[:-4]), prob = label_prob, min_prob = label_prob_min)#std = std_out ) #, neg = out_neg)

        #del final_output, final_output_negative, final_output_k, final_output_negative_k, std_out, out_neg, out_neg_1, out_neg_2, label, label_std
    #print('the total negative label percentage is: ', neg_percent)
    print(args.restore_from)
    print('the number of positive labels that have less than 80% confidence are: ', posi_ignore)
    print('the number of positive labels (chosen by minimum) that have less than 80% confidence are: ', posi_ignore_min)
    print('the number of picture in train is: ', pic_count)
    return None

def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--num-classes", type=int, default=9,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--mc-dropout", type=bool, default=True,
                        help="whether to use mc-dropout in generating pseudo labels.")
    parser.add_argument("--data-dir", type=str, default='./data/Oxford_Robot_ICCV19/',
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default='./dataset/robot_list/train.txt',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--temp-scale", type=int, default=2,
                        help="the temperature scaling factor for negative labelling")
    parser.add_argument("--set", type=str, default='train',
                        help="choose evaluation set.")
    parser.add_argument("--restore-from", type=str, default='./transition/GTA/snapshots/strategy_5/GTA_25000.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--save-path", type=str, default='./transition/GTA/pseudo/3/',
                        help="path to save the pseudo labels.")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    restore_from = args.restore_from
    gen_pn_plabel(args, restore_from)