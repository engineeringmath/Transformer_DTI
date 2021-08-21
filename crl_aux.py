#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:25:24 2019

@author: ch209389
"""


import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
#import numpy.polynomial.polynomial as poly
#import sys
# from tqdm import tqdm
import SimpleITK as sitk
# from scipy.interpolate import SmoothSphereBivariateSpline
# from scipy.spatial import ConvexHull
# import crl_dci
# import math
# from dipy.direction.peaks import peak_directions
# from scipy.stats import entropy
# import dipy.reconst.shm as shm
# import time





np.warnings.filterwarnings('ignore')






def optimized_six():
    
    p= 0.910
    q= (1-p**2)**0.5
    
    xp= np.array( [ p,    p  ,  0.0,  0.0 ,   q,    -q  ] )
    yp= np.array( [ q,   -q  ,   p ,   p  ,  0.0,   0.0 ] )
    zp= np.array( [ 0.0, 0.0 ,  -q ,   q  ,   p ,    p  ] )
    
    return xp, yp, zp





def seg_2_boundary_3d(x):
    
    a, b, c= x.shape
    
    y= np.zeros(x.shape)
    z= np.nonzero(x)
    
    if len(z[0])>1:
        x_sum= np.zeros(x.shape)
        for shift_x in range(-1, 2):
            for shift_y in range(-1, 2):
                for shift_z in range(-1, 2):
                    x_sum[1:-1,1:-1,1:-1]+= x[1+shift_x:a-1+shift_x, 1+shift_y:b-1+shift_y, 1+shift_z:c-1+shift_z]
        y= np.logical_and( x==1 , np.logical_and( x_sum>0, x_sum<27 ) )

    return y





def skull_from_brain_mask(brain_mask, radius= 2.0):
    
    mask_copy= brain_mask.copy()
    
    size_x, size_y, size_z= brain_mask.shape
    mask= np.zeros((size_x+20, size_y+20, size_z+20))
    mask[10:10+size_x, 10:10+size_y, 10:10+size_z]= mask_copy
    
    mask_boundary= seg_2_boundary_3d(mask)
    
    mask_boundary= sitk.GetImageFromArray(mask_boundary.astype(np.int))
    
    dist_image = sitk.SignedMaurerDistanceMap(mask_boundary, insideIsPositive=True, useImageSpacing=True,
                                           squaredDistance=False)
    
    dist_image= - sitk.GetArrayFromImage(dist_image)
    dist_image= dist_image<radius
    
    skull= dist_image * mask
    
    '''mask= brain_mask.copy()
    
    mask_boundary= seg_2_boundary_3d(mask)
    
    mask_boundary= sitk.GetImageFromArray(mask_boundary.astype(np.int))
    
    dist_image = sitk.SignedMaurerDistanceMap(mask_boundary, insideIsPositive=True, useImageSpacing=True,
                                           squaredDistance=False)
    
    dist_image= - sitk.GetArrayFromImage(dist_image)
    dist_image= dist_image<radius
    
    skull= dist_image * mask'''
    
    skull= skull[10:10+size_x, 10:10+size_y, 10:10+size_z]
    
    return skull






def find_closest_bvecs_no_rotation(v, b_vecs, antipodal=False):
    
    if antipodal:
        temp= np.abs(np.dot(b_vecs, v))
    else:
        temp= np.dot(b_vecs, v)
    
    angs= np.arccos( np.clip( temp, 0, 1) )*180/np.pi
    
    ind_bvecs= np.zeros(v.shape[1], dtype=np.int)
    ang_bvecs= np.zeros(v.shape[1])
    
    for i in range(v.shape[1]):
        
        ang_min= angs.min()
        temp= np.where(angs==ang_min)
        indx, indy= temp[0][0], temp[1][0]
        
        ind_bvecs[indy]= indx
        ang_bvecs[indy]= ang_min
        
        angs[:,indy]= 180
    
    return ind_bvecs, ang_bvecs, v





