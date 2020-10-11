# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 19:33:46 2020

@author: 吴九龙

Input:Images must be numbered from left to right
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

MIN = 10 #匹配点的最少数量
img_number = 3 #输入的图片的数量
img_type = 'jpg' #输入的图片的格式
iterations = img_number - 1 #迭代次数
img_path = './images/'

def brightness(img1,img2):
    # 输入图像亮度差别太大的时候用以平均亮度
    img1_hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    h1,s1,v1 = cv2.split(img1_hsv)
    
    img2_hsv = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
    h2,s2,v2 = cv2.split(img2_hsv)
    
    # 平均图像亮度
    v = cv2.subtract(v1,v2)
    v = cv2.divide(v,2)
    v = np.abs(v)
    
    p = np.mean(v) 
    v1 = cv2.add(1*v1,p)
    v2 = cv2.add(1*v2,p)
    
    img1 = np.uint8(cv2.merge((h1,s1,v1)))
    img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2BGR)
    
    img2 = np.uint8(cv2.merge((h2,s2,v2)))
    img2 = cv2.cvtColor(img2,cv2.COLOR_HSV2BGR)
    
    return img1,img2

def reshape(img1,img2):
    # 输入图像大小不统一时用reshape统一图像shape
    r1,c1 = img1.shape[:2]
    r2,c2 = img2.shape[:2]
    row = max(r1,r2)
    col = max(c1,c2)
    
    # 固定图像大小
    arr1 = np.zeros([row,col,3])
    arr2 = np.zeros([row,col,3])
    
    for i in range(r1):
        for j in range(c1):
            arr1[i,j] = img1[i,j]
    for i in range(r2):
        for j in range(c2):
            arr2[i,j] = img2[i,j]
            
    return np.uint8(arr1),np.uint8(arr2)

def merge(img1,img2,itera):
    # 两张图片拼接的函数，用循环实现多图像拼接
    img1,img2 = reshape(img1,img2)
#    img1,img2 = brightness(img1,img2)
    img1_rgb,img2_rgb = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB),cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    plt.imshow(img1_rgb),plt.show()
    plt.imshow(img2_rgb),plt.show()
    
    #img1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    surf=cv2.xfeatures2d.SURF_create(400,nOctaves=4,extended=False,upright=True)
    #surf=cv2.xfeatures2d.SIFT_create()#可以改为SIFT
    kp1,descrip1=surf.detectAndCompute(img1,None)
    kp2,descrip2=surf.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    searchParams = dict(checks=50)
    
    flann=cv2.FlannBasedMatcher(indexParams,searchParams)
    match=flann.knnMatch(descrip1,descrip2,k=2)
    
    
    good=[]
    for i,(m,n) in enumerate(match):
            if(m.distance<0.75*n.distance):
                    good.append(m)
                    
    # cv2.drawMatchesKnn expects list of lists as matches.
    good_2 = np.expand_dims(good, 1)
    matching = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_2[:20],None, flags=2)
    cv2.imwrite(img_path+"matching{}.png".format(itera),matching)
    
    matching=cv2.cvtColor(matching,cv2.COLOR_BGR2RGB)
    plt.imshow(matching,),plt.show()
    
    print('similar anchours:',len(good))
    
    if len(good)>MIN:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M,mask=cv2.findHomography(src_pts,ano_pts,cv2.RANSAC,5.0)
            warpImg = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1]+img2.shape[1], img2.shape[0]))
            direct = warpImg.copy()
            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    if img1[i,j].any():
                        direct[i,j] = img1[i,j]
            simple = time.time()
            
            rows,cols=img1.shape[:2]
            
            print('Step1 warp finished')
            
            for col in range(0,cols):
                if img1[:, col].any() and warpImg[:, col].any():#开始重叠的最左端
                    left = col
                    break
            for col in range(cols-1, 0, -1):
                if img1[:, col].any() and warpImg[:, col].any():#重叠的最右一列
                    right = col
                    break
    
            res = np.zeros([rows, cols, 3], np.uint8)
            for row in range(0, rows):
                for col in range(0, cols):
                    if not img1[row, col].any():#如果没有原图，用旋转的填充
                        res[row, col] = warpImg[row, col]
                    elif not warpImg[row, col].any():
                        res[row, col] = img1[row, col]
                    else:
                        srcImgLen = float(abs(col - left))
                        testImgLen = float(abs(col - right))
                        alpha = srcImgLen / (srcImgLen + testImgLen)
                        res[row, col] = np.clip(img1[row, col] * (1-alpha) + warpImg[row, col] * alpha, 0, 255)
    
            warpImg[0:img1.shape[0], 0:img1.shape[1]]=res
            iteration=time.time()
            
            rows,cols=warpImg.shape[:2]
            
            for col in range(0,cols):
                if warpImg[:, col].any():#图像的最左端
                    left = col
                    break
            for col in range(cols-1, 0, -1):
                if warpImg[:, col].any():#图像的最右一列
                    right = col
                    break
            
            warpImg = warpImg[:,left:right+1,:]
            
            print('Step2 merge finished')
            
            img3=cv2.cvtColor(direct,cv2.COLOR_BGR2RGB)
            plt.imshow(img3,),plt.show()
            img4=cv2.cvtColor(warpImg,cv2.COLOR_BGR2RGB)
            plt.imshow(img4,),plt.show()
            print("simple stich cost %f"%(simple-starttime))
            print("\niteration cost %f"%(iteration-starttime))
            cv2.imwrite(img_path+"simplepanorma{}.png".format(itera),direct)
            cv2.imwrite(img_path+"bestpanorma{}.png".format(itera),warpImg)
            
            return warpImg
    else:
            print("not enough matches!")


if __name__ == '__main__':
    starttime=time.time() 
    img1 = cv2.imread(img_path+'img1.{}'.format(img_type)) #query
    
    for i in range(1,iterations+1):
        print('Iteration number:{}'.format(i))
        img2 = cv2.imread(img_path+'img{}.{}'.format(i+1,img_type))
        img1 = merge(img1,img2,i)
                
    final = time.time()
    print("\nfinal cost %f"%(final-starttime))
