# -*- coding: utf-8 -*-
"""
"""


from __future__ import division

# import numpy as np
import SimpleITK as sitk
#import os
# from scipy.spatial.distance import directed_hausdorff
# from scipy.spatial.distance import cdist







def resample3d(x, original_spacing, new_spacing, resampling_mode):
    '''Resamples the given array into new spacing'''
    
    original_size = x.GetSize()
    I_size = [int(spacing/new_s*size) 
                            for spacing, size, new_s in zip(original_spacing, original_size, new_spacing)] 
    
    I = sitk.Image(I_size, x.GetPixelIDValue())
    I.SetSpacing(new_spacing)
    I.SetOrigin(x.GetOrigin())
    I.SetDirection(x.GetDirection())
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(I)
    resample.SetInterpolator(resampling_mode)
    resample.SetTransform(sitk.Transform())
    I = resample.Execute(x)
    
    return I



