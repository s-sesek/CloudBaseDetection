#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:22:19 2019

@author: sarah
"""
import cv2
import math
#import matplotlib.pyplot as plt
import numpy as np
import os

def main():
  list1=["9000mask110", "10800mask079", "12600mask013", "14400mask033", "16200mask019", "18000mask077", "19800mask041", "21600mask062", "37800mask031", "39600mask063", "41400mask063", "41400mask108"]
  os.chdir("/home/sarah/Downloads")
  white =np.array([231,231,231])
  gray = np.array([103,103,103])
  black = np.array([0,0,0])
  green = np.array([1,231, 95])
  for j in list1:
    img = cv2.imread(j + ".png")
    for x in range(0, 319):
      for y in range(0,239):
        dw = calcDistance(img[x][y], white)
        db = calcDistance(img[x][y], black)
        dgn = calcDistance(img[x][y], green)
        dgy = calcDistance(img[x][y], gray)
        closeColor = min([dw, db, dgn, dgy])
        if(closeColor==db):
          convertColor(img[x][y], black)
        elif(closeColor==dgn):
          convertColor(img[x][y], green)
        elif(closeColor==dw):
          convertColor(img[x][y], white)
        else:
          convertColor(img[x][y], gray)
         
    name = j+"fixed.png"
    cv2.imwrite(name, img)

def calcDistance(pixel, color):
  distance=math.sqrt((pixel[0]-color[0])**2+(pixel[1]-color[1])**2+(pixel[2]-color[2])**2)
  return distance;

def convertColor(pixel, color):
  pixel[0]=color[0]
  pixel[1]=color[1]
  pixel[2]=color[2]
      
main()