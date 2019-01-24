#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:41:29 2018

@author: sarah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:18:33 2018

@author: nick
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

'''
User options: 
    Masked  = True 
        returns masked images in directory
    Masked = False 
        returns 'pretty' images
    
    Directory = 'string' 
        please modify this string to the parent directory of the videos
        
    Name = 'string'
        Modify to correspond to the name of the dataset
    Time = int
        Simulation time of the image (hours please)
'''
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('Masked', None, 'Returns masked images.')
flags.DEFINE_string('video_directory', None, 'Directory with videos')
flags.DEFINE_string('folder', None, ' ')


#Name = 'LASSO 20160611 '
#Time = 28800/3600

'''
Source
 |
 |
 V
'''
xcoords = [6400, 9600, 12800, 16000]#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:41:29 2018

@author: sarah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:18:33 2018

@author: nick
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

'''
User options: 
    Masked  = True 
        returns masked images in directory
    Masked = False 
        returns 'pretty' images
    
    Directory = 'string' 
        please modify this string to the parent directory of the videos
        
    Name = 'string'
        Modify to correspond to the name of the dataset
    Time = int
        Simulation time of the image (hours please)
'''
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('Masked', None, 'Returns masked images.')
flags.DEFINE_string('video_directory', None, 'Directory with videos')
flags.DEFINE_string('folder', None, ' ')


#Name = 'LASSO 20160611 '
#Time = 28800/3600

'''
Source
 |
 |
 V
'''
xcoords = [6400, 9600, 12800, 16000, 19200]

newpath = r'/data/scripts/LES_TSI/SmallData/targets/43200'

if not os.path.exists(newpath):
	os.makedirs(newpath)
    	
if FLAGS.Masked == True:
	name_0 = 'TSI_R '
	os.chdir(newpath)
else:
    name_0 = 'TSI_R_MASKED '
    os.chdir(newpath)

for xcoord in xcoords:
    name_1 = '(along x = '+str(xcoord)+').avi'
    video = cv2.VideoCapture(FLAGS.video_directory+name_0+name_1)
    
    i = -1# framenumber
    j = -1
    success = True
    while success == True:
        i += 1
        j += 1
        ycoord = i * 150 #m 
        plt.figure()
        success, img = video.read()
        if img is None:
            raise NameError('Files are not in Directory, rename directory to reflect correct directory')
            break
        '''
        Several options may exist at this point, simply saving the data or 
        utilizing some sort of unwrapping algorithm. Or, at this point, the 
        image may be fed to the machine learning algorithm
        '''
        
        #plt.title('TSI_Render at: x,y = '+str(xcoord)+', '+str(ycoord)+'(m)')
        img2 = np.uint8(img)
        if FLAGS.Masked == True:
        	if -1 < i <= 9:
        		string = FLAGS.folder + 'frame00' + str(i)+ '.png'
        	if 10 <= i <= 99:
        		string = FLAGS.folder + 'frame0' + str(i)+ '.png'
        	if 100 <= i:
        		string = FLAGS.folder + 'frame' + str(i)+ '.png'
        else:
        	if -1 < i <= 9:
        		string = FLAGS.folder + 'mask00' + str(i)+ '.png'
        	if 10 <= i <= 99:
        		string = FLAGS.folder + 'mask0' + str(i)+ '.png'
        	if 100 <= i:
        		string = FLAGS.folder + 'mask' + str(i)+ '.png'
        #  string = FLAGS.folder + 'mask' + str(i)+ '.png'	
        cv2.imwrite(string, img2)
        #plt.imshow(img2)
        #plt.imshow(np.array(img))
        #plt.show()
        

if FLAGS.Masked == False:
    name_0 = 'TSI_R '
    os.chdir('/data/scripts/LES_TSI/SmallData/inputs/')
else:
    name_0 = 'TSI_R_MASKED '
    os.chdir('/data/scripts/LES_TSI/SmallData/targets/')

for xcoord in xcoords:
    name_1 = '(along x = '+str(xcoord)+').avi'
    video = cv2.VideoCapture(FLAGS.video_directory+name_0+name_1)
    
    i = -1# framenumber
    success = True
    while success == True:
        i += 1
        ycoord = i * 150 #m 
        plt.figure()
        success, img = video.read()
        if img is None:
            raise NameError('Files are not in Directory, rename directory to reflect correct directory')
            break
        '''
        Several options may exist at this point, simply saving the data or 
        utilizing some sort of unwrapping algorithm. Or, at this point, the 
        image may be fed to the machine learning algorithm
        '''
        
        #plt.title('TSI_Render at: x,y = '+str(xcoord)+', '+str(ycoord)+'(m)')
        img2 = np.uint8(img)
        if FLAGS.Masked == False:
          string = FLAGS.folder + 'frame' + str(i)+ '.png'
        else:
          string = FLAGS.folder + 'mask' + str(i)+ '.png'
        cv2.imwrite(string, img2)
        #plt.imshow(img2)
        #plt.imshow(np.array(img))
        #plt.show()
        