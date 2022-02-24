import os
import os.path as osp
import numpy as np
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image,ImageFile
from dataset.autoaugment import ImageNetPolicy
import torch.nn as nn

ImageFile.LOAD_TRUNCATED_IMAGES = True

class cityscapes_plabel_dataset(data.Dataset):
    def __init__(self, root, label_path, list_path, max_iters=None, resize_size=(1024, 512), crop_size=(512, 1024), mean=(128, 128, 128), scale=False, mirror=True, ignore_label=255, set='val', autoaug=False):
        self.root = root
        self.label_path = label_path
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.resize_size = resize_size
        self.autoaug = autoaug
        self.h = crop_size[0]
        self.w = crop_size[1]
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
         
        #https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.label_path, name)
            name2 = name[:-4] + '_2.png'
            label2_file = osp.join(self.label_path, name2)
            other_file = osp.join(self.label_path, (name[:-3]+'npz'))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "label_2": label2_file,
                "other": other_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label_std = Image.open(datafiles["label_2"])
        other = np.load(datafiles['other'])
        std = other['std']
        neg_label = other['neg'].astype(float)
        neg_label = neg_label.transpose(2, 0, 1)
        name = datafiles["name"]


        #str_print = 'std shape at the beginning' + str(std.shape)
        #str_print += '\n neg label shape at the beginning' + str(neg_label.shape)

        del other
        std = torch.from_numpy(std)
        neg_label = torch.from_numpy(neg_label)
        std = std.unsqueeze(0).unsqueeze(0)
        neg_label = neg_label.unsqueeze(0)

        #str_print += '\n std shape after adding channel and batchsize' + str(std.shape)
        #str_print += '\n neg shape after adding channel and batchsize' + str(neg_label.shape)

        # resize
        if self.scale:
            random_scale = 0.8 + random.random()*0.4 # 0.8 - 1.2
            size1 = round(self.resize_size[0] * random_scale)
            size2 = round(self.resize_size[1] * random_scale)
            image = image.resize( (size1, size2) , Image.BICUBIC)
            label = label.resize( (size1, size2), Image.NEAREST)
            label_std = label_std.resize((size1, size2), Image.NEAREST)
            #label_negative = label_negative.resize( ( round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)) , Image.NEAREST)

            std = nn.functional.interpolate(std, size=[size2, size1], mode='bilinear', align_corners=True)
            neg_label = nn.functional.interpolate(neg_label, size=[size2, size1], mode='nearest')

        else: #this is not going to happen
            image = image.resize( ( self.resize_size[0], self.resize_size[1] ) , Image.BICUBIC)
            #label = label.resize( ( self.resize_size[0], self.resize_size[1] ) , Image.NEAREST) #they have already been in this shape, so don't need to be resized
            #label_negative = label_negative.resize( ( self.resize_size[0], self.resize_size[1] ) , Image.NEAREST)

        if self.autoaug:
            policy = ImageNetPolicy()
            image = policy(image)



        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.uint8)
        label_std = np.asarray(label_std, np.uint8)

        std = std.squeeze(0).squeeze(0)
        std = std.numpy()
        neg_label = neg_label.squeeze(0)
        neg_label = neg_label.numpy()
        #neg_label = neg_label.transpose(1, 2, 0)
        #label_negative = np.asarray(label_negative, np.uint8)


        # re-assign labels to match the format of Cityscapes
        #label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        #for k, v in list(self.id_to_trainid.items()):
        #    label_copy[label == k] = v
        #label_copy = label
        #label_copy_negative = label_negative

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        #print(image.shape, label.shape)
        #str_print += '           image shape after resize' + str(image.shape)
        #str_print += '           label shape after resize' + str(label.shape)
        #str_print += '           std shape after resize' + str(std.shape)
        #str_print += '           neg label shape after resize and transpose'+ str(neg_label.shape)

        for i in range(10): #find hard samples
            x1 = random.randint(0, image.shape[1] - self.h)
            y1 = random.randint(0, image.shape[2] - self.w)
            tmp_label = label[x1:x1+self.h, y1:y1+self.w]
            tmp_label_std = label_std[x1:x1+self.h, y1:y1+self.w]
            tmp_std = std[x1:x1+self.h, y1:y1+self.w]
            tmp_neg = neg_label[:,x1:x1+self.h, y1:y1+self.w]

            tmp_image = image[:, x1:x1+self.h, y1:y1+self.w]
            u = np.unique(tmp_label)
            if len(u) > 10:
                break
            #else:
                #print('Cityscape-Pseudo: Too young too naive for %d times!'%i)
        image = tmp_image
        label = tmp_label
        label_std = tmp_label_std
        std = tmp_std
        neg_label = tmp_neg

        #str_print += '           image shape after crop' + str(image.shape)
        #str_print += '           label shape after crop' + str(label.shape)
        #str_print+='          std shape after crop' + str(std.shape)
        #str_print+='          neg label shape after crop' +  str(neg_label.shape)

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis = 2)
            label = np.flip(label, axis = 1)
            label_std = np.flip(label_std, axis = 1)
            std = np.flip(std, axis=1)
            neg_label = np.flip(neg_label, axis = 2)


        #str_print+='         std shape after flip' + str(std.shape)
        #str_print += '         neg label shape' + str(neg_label.shape)

        return image.copy(), label.copy(), std.copy(), neg_label.copy(), label_std.copy(), np.array(size), name

        #return image.copy(), label.copy(), std.copy(), neg_label.copy(), label_std.copy(), str_print, name
