#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import pdb
import sys
import cv2
import numpy as np
import os.path as osp
from glob import glob
from pycocotools import mask
from skimage import measure
import argparse
from tqdm import tqdm

root_path = '/scratch/qizhicai/daggerR6maskrcnn/images'  # image size 352*640 id.png, label.png, final.jpg



def generate_id_map(map_path):
    id_map = cv2.imread(map_path, -1)
    h, w, _ = id_map.shape
    id_map = np.concatenate((id_map, np.zeros((h, w, 1), dtype=np.uint8)), axis=2)
    id_map.dtype = np.uint32
    return id_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform gta to coco')
    parser.add_argument('--phase', type=str, default='dev', help='phase')
    args = parser.parse_args()
    phase = args.phase
    dataset = {
        'licenses': [],
        'info': {},
        'categories': [],
        'images': [],
        'annotations': []
    }

    classes = ['person', 'car']

    for i, cls in enumerate(classes):
        dataset['categories'].append({
            'id': i+1,
            'name': cls,
            'supercategory': 'object'
        })

    indexes = sorted(glob(osp.join(root_path, 'rec*', '*_final.jpg')))
    num_img = len(indexes)

    if phase == 'train':
        indexes = indexes[:int(num_img * 0.9)]
    elif phase == 'val':
        indexes = indexes[int(num_img * 0.9):]
    elif phase == 'dev':
        indexes = indexes[0:1000:5]

    for id, imgpath in enumerate(tqdm(indexes)):
        if not(osp.exists(imgpath.replace('final.jpg', 'id.png')) and osp.exists(imgpath.replace('final.jpg', 'label.png'))):
            continue
        dataset['images'].append({
            'coco_url': '',
            'date_captured': '',
            'file_name': osp.relpath(imgpath, root_path),
            'flickr_url': '',
            'id': id,
            'license': 0,
            'width': 640,
            'height': 352
        })
        id_map = generate_id_map(imgpath.replace('final.jpg', 'id.png'))[:,:,0]
        label_map = cv2.imread(imgpath.replace('final.jpg', 'label.png'))[:,:,0]
        if not (2 in np.unique(label_map) or 1 in np.unique(label_map)):
            label_map_ = np.zeros_like(label_map)
            label_map_[id_map != 0] = 2
            label_map_[label_map == 60] = 3
            label_map_[label_map == 142] = 3
            label_map_[label_map == 180] = 1
            label_map_[label_map == 152] = 1
            label_map = label_map_
        instanceset = np.unique(id_map).tolist()
        ego_id = id_map[351, 640//2]
        instanceset.remove(0)
        if ego_id!=0:
            instanceset.remove(ego_id)
        for instance in instanceset:
            mask = (id_map==instance)
            pixels = np.sum(mask)
            if pixels<50:
                continue
            cls = label_map[mask][0]
            if cls == 0 or cls == 3:
                continue
            y_list, x_list = np.where(id_map==instance)
            bbox = [
                np.amin(x_list),
                np.amin(y_list),
                np.amax(x_list),
                np.amax(y_list)]  # x1, y1, x2, y2
            x1, y1, x2, y2 = bbox
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            contours = measure.find_contours(mask.astype(np.uint8), 0.5)
            seg = []
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                seg.append(segmentation)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(cls),
                'id': instance,
                'image_id': id,
                'iscrowd': 0,
                'segmentation': seg
            })

    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
    with open(json_name, 'w') as f:
        json.dump(dataset, f)
