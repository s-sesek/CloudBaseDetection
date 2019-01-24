# Copyright 2018 Edward Allums, Gopal Godhani, Dan Mayich, and Sarah Sesek
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:27:57 2018

@author: sarah
"""
import cv2
#import numpy as np
import tensorflow as tf
import os

all_targets_folder='/data/scripts/LES_TSI/SmallData/targets/'

def convert_blues(mask_name, filename):
    mask = cv2.imread(mask_name)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            k=mask[x,y]
            if k[0]!=0 and k[1]!=0 and k[2]!=0:
                if k[0]!=231 and k[1]!=231 and k[2]!=231: 
                    if k[0]!=103 and k[1]!=103 and k[2]!=103:
                        k[0]=255
                        k[1]=191
                        k[2]=0
    cv2.imwrite(all_targets_folder+filename, mask)                    
                        
seg_names = tf.gfile.Glob(os.path.join(all_targets_folder, '*.png'))#change from .jpg
base_names=[]
for i in range(len(seg_names)):
    base_names.append(os.path.basename(seg_names[i]))
    convert_blues(seg_names[i], base_names[i])
