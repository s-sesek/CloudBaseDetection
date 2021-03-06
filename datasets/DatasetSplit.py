#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:04:53 2018

@author: sarah
"""
import shutil
import os
from math import ceil
import random
import tensorflow as tf


all_inputs_folder='/data/scripts/LES_TSI/SmallData/inputs/'
all_targets_folder='/data/scripts/LES_TSI/SmallData/targets/'

img_names = tf.gfile.Glob(os.path.join(all_inputs_folder, '*.png'))#change from .jpg
random.shuffle(img_names)
if len(img_names)!=0:
    for i in range(len(img_names)):
        # get the filename without the extension
        img_names[i] = os.path.basename(img_names[i]).split('.')[0]                              #change to include randomly choosong val/test set
        if i <= int(ceil(len(img_names)*.7)):
            shutil.move(all_inputs_folder+img_names[i]+".png", 
                        all_inputs_folder+"train/"+img_names[i]+".png")
        else:
            shutil.move(all_inputs_folder+img_names[i]+".png", 
                        all_inputs_folder+"test/"+img_names[i]+".png")
seg_names = []
for f in range(len(img_names)):
    # cover its corresponding *_seg.png
    target_name=img_names[f].replace("frame", "mask")
    seg_names.append(target_name)
for g in range(len(seg_names)):
    if g <= int(ceil(len(seg_names)*.7)):
        shutil.move(all_targets_folder+seg_names[g]+".png", 
                    all_targets_folder+"train/"+seg_names[g]+".png")
    else:
        shutil.move(all_targets_folder+seg_names[g]+".png",
                    all_targets_folder+"test/"+seg_names[g]+".png")          
  