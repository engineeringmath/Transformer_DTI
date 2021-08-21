#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 10:29:03 2020

@author: davood
"""




import numpy as np
# import crl_dci
import crl_aux
# import matplotlib.pyplot as plt
from dipy.data import get_sphere
#from sklearn import svm
import tensorflow as tf
import dk_model
import os
# from tqdm import tqdm
# import h5py
#from numpy import matlib as mb
from os import listdir
from os.path import isfile, join
import dipy.core.sphere as dipysphere
# from dipy.reconst.shm import sf_to_sh
# from dipy.reconst.shm import sh_to_sf
# import dk_aux
# import numpy as np
# from os import listdir
# from os.path import isdir, join
#from dipy.io.image import load_nifti, save_nifti
#from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
#from dipy.reconst.dti import TensorModel
#from dipy.data import fetch_sherbrooke_3shell
#from os.path import expanduser, join
#import nibabel as nib
# import matplotlib.pyplot as plt
#from dipy.io import read_bvals_bvecs
#from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
import SimpleITK as sitk
import nibabel as nib
#import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa, mean_diffusivity
import crl_dti
# import crl_dci
#import scipy.optimize as opt
# from dipy.data import get_sphere
# import dk_aux
# import crl_aux
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
# from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.core.geometry import vector_norm
# import tensorflow as tf
# import dk_model
# import os
# from tqdm import tqdm
# try:
#     from fury import actor, window
# except:
#     print('No fury!')
# import matplotlib.patches as patches
# from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
# from dmipy.signal_models import cylinder_models, gaussian_models
# from dmipy.core.modeling_framework import MultiCompartmentModel
# import dmipy.core.acquisition_scheme as dmipyschm
# import scipy.optimize as opt
from dipy.direction import ClosestPeakDirectionGetter
# from dipy.direction import ProbabilisticDirectionGetter
# from dipy.io.streamline import load_trk
# from dipy.tracking.utils import length
# from dipy.direction.peaks import peak_directions
# import h5py
# from dipy.reconst.shm import sf_to_sh
# from dipy.reconst.shm import sh_to_sf
# import pandas as pd
# import dipy.reconst.shm as shm
# import dipy.direction.peaks as dp
# from dipy.segment.tissue import TissueClassifierHMRF
import dk_seg
# from scipy import ndimage
# from sklearn.neighbors import KernelDensity



MODEL= 'Transformer'
MODEL= 'DTI_Transformer_plusplus'

MODEL= 'CNN'



N_grad= 6
Xp, Yp, Zp= crl_aux.optimized_six()
sphere_grad = dipysphere.Sphere(Xp, Yp, Zp)
v_grad, _ = sphere_grad.vertices, sphere_grad.faces

sphere_odf = get_sphere('repulsion724')
v_odf, _ = sphere_odf.vertices, sphere_odf.faces


gpu_ind= 1
# L_Rate = 5.0e-4



n_channel= N_grad
n_class=   6

SX, SY, SZ= 125, 125, 80



if MODEL== 'Transformer':
    
    L= 5
    residuals=  True
    use_token=  False
    layer_norm= False
    learn_embd= True
    d_emb_h= 512
    n_head= 2
    d_emb=  d_emb_h*n_head
    ViT_depth= 4
    L_Rate = 1.0e-4
    w_std=   0.2
    n_channel= N_grad
    n_inp= L**3
    
    X = tf.placeholder("float32", [None, n_inp, n_channel ])
    T = tf.placeholder("float32", [None, 1, d_emb ])
    Y = tf.placeholder("float32", [None, L, L, L, n_class ])
    pos_emb = tf.Variable(tf.truncated_normal([n_inp, d_emb], stddev= 0.1), name='pos_emb')
    
    learning_rate = tf.placeholder("float")
    p_keep = tf.placeholder("float")
    
    Y_pred, A_list = dk_model.DTI_Transformer(X, T, pos_emb, n_inp, d_emb, ViT_depth, n_head, d_emb_h, n_channel, L, n_class=n_class, 
                           use_token=use_token, residuals=residuals, layer_norm=layer_norm, learn_embd=learn_embd, p_keep=p_keep, w_std=w_std)
    
    LX= LY= LZ= L
    test_shift= LX//2
    lx_list= np.squeeze( np.concatenate(  (np.arange(0, SX-LX, test_shift)[:,np.newaxis] , np.array([SX-LX])[:,np.newaxis] )  ) .astype(np.int) )
    ly_list= np.squeeze( np.concatenate(  (np.arange(0, SY-LY, test_shift)[:,np.newaxis] , np.array([SY-LY])[:,np.newaxis] )  ) .astype(np.int) )
    lz_list= np.squeeze( np.concatenate(  (np.arange(0, SZ-LZ, test_shift)[:,np.newaxis] , np.array([SZ-LZ])[:,np.newaxis] )  ) .astype(np.int) )
    LXc= LYc= LZc= 0# LX//4

elif MODEL=='DTI_Transformer_plusplus':
    
    L= 5
    residuals=  True
    use_token=  False
    layer_norm= False
    learn_embd= True
    d_emb_h= 512
    n_head= 2
    d_emb=  d_emb_h*n_head
    ViT_depth= 4
    L_Rate = 1.0e-4
    w_std=   0.2
    n_channel= N_grad
    n_inp= L**3
    
    X = tf.placeholder("float32", [None, n_inp, n_channel ])
    T = tf.placeholder("float32", [None, 1, d_emb ])
    Y = tf.placeholder("float32", [None, L, L, L, n_class ])
    pos_emb = tf.Variable(tf.truncated_normal([n_inp, d_emb], stddev= 0.1), name='pos_emb')
    Dpos_emb = tf.Variable(tf.truncated_normal([n_inp, d_emb], stddev= 0.1), name='pos_emb')
    Spos_emb = tf.Variable(tf.truncated_normal([n_inp, d_emb], stddev= 0.1), name='pos_emb')
    
    learning_rate = tf.placeholder("float")
    p_keep = tf.placeholder("float")
    
    Y_pred, A_list = dk_model.DTI_Transformer_plusplus(X, T, pos_emb, Spos_emb, Dpos_emb, n_inp, d_emb, ViT_depth, n_head, d_emb_h, n_channel, L, n_class=n_class, 
                               use_token=use_token, residuals=residuals, layer_norm=layer_norm, learn_embd=learn_embd, p_keep=p_keep, w_std=w_std)
    
    cost= tf.reduce_mean( tf.pow( Y_pred- Y, 2 ) )
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    LX= LY= LZ= L
    test_shift= LX//2
    lx_list= np.squeeze( np.concatenate(  (np.arange(0, SX-LX, test_shift)[:,np.newaxis] , np.array([SX-LX])[:,np.newaxis] )  ) .astype(np.int) )
    ly_list= np.squeeze( np.concatenate(  (np.arange(0, SY-LY, test_shift)[:,np.newaxis] , np.array([SY-LY])[:,np.newaxis] )  ) .astype(np.int) )
    lz_list= np.squeeze( np.concatenate(  (np.arange(0, SZ-LZ, test_shift)[:,np.newaxis] , np.array([SZ-LZ])[:,np.newaxis] )  ) .astype(np.int) )
    LXc= LYc= LZc= 0# LX//4
    
elif MODEL== 'CNN':
    
    LX, LY, LZ = 32, 32, 32
    test_shift= LX//2
    n_feat_0 = 14
    depth = 4
    ks_0 = 3
    
    X = tf.placeholder("float32", [None, LX, LY, LZ, n_channel])
    Y = tf.placeholder("float32", [None, LX, LY, LZ, n_class])
    learning_rate = tf.placeholder("float")
    p_keep_conv = tf.placeholder("float")
    
    Y_pred, U_pred = dk_model.davood_net(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001)
    
    test_shift= LX//4
    lx_list= np.squeeze( np.concatenate(  (np.arange(0, SX-LX, test_shift)[:,np.newaxis] , np.array([SX-LX])[:,np.newaxis] )  ) .astype(np.int) )
    ly_list= np.squeeze( np.concatenate(  (np.arange(0, SY-LY, test_shift)[:,np.newaxis] , np.array([SY-LY])[:,np.newaxis] )  ) .astype(np.int) )
    lz_list= np.squeeze( np.concatenate(  (np.arange(0, SZ-LZ, test_shift)[:,np.newaxis] , np.array([SZ-LZ])[:,np.newaxis] )  ) .astype(np.int) )
    LXc, LYc, LZc= 12, 12, 12






os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
saver = tf.train.Saver(max_to_keep=1000)
sess = tf.Session()
sess.run(tf.global_variables_initializer())





if MODEL== 'Transformer' or MODEL=='DTI_Transformer_plusplus':
    
    restore_model_path= '/src/model/Trnasformer/model_000046_1200.ckpt'
    saver.restore(sess, restore_model_path)

elif MODEL== 'CNN':
    
    restore_model_path= '/src/model/CNN/model_000015_625.ckpt'
    saver.restore(sess, restore_model_path)






















test_dir= '/src/test_images/'

results_dir= test_dir + 'tensors/'
os.makedirs( results_dir , exist_ok=True)




img_files = [f for f in listdir(test_dir) if isfile( join(test_dir, f) ) and 'dwi.nii.gz' in f]
mask_files = [f for f in listdir(test_dir) if isfile( join(test_dir, f) ) and 'brainmask' in f]

bval_files = [f for f in listdir(test_dir) if isfile( join(test_dir, f) ) and 'bval' in f]
bvec_files = [f for f in listdir(test_dir) if isfile( join(test_dir, f) ) and 'bvec' in f]

img_files.sort()





n_cases= len(img_files)


desired_spacing= (1.2, 1.2, 1.2)
sx, sy, sz= 125, 125, 80
keep_test= 1.0



for i_file in range(n_cases):
    
    print('Running the segmentation on ', img_files[i_file] )
    
    file_name= test_dir+ img_files[i_file]
    d_img= sitk.ReadImage( file_name )
    d_img= sitk.GetArrayFromImage(d_img)
    d_img= np.transpose( d_img, [3,2,1,0] )
    
    b_vals= np.loadtxt( test_dir+ bval_files[i_file] )
    b_vecs= np.loadtxt( test_dir+ bvec_files[i_file] )
    
    file_name= test_dir+ mask_files[i_file]
    brain_mask_img= sitk.ReadImage( file_name )
    brain_mask= sitk.GetArrayFromImage(brain_mask_img)
    brain_mask= np.transpose( brain_mask, [2,1,0] )
    
    #############
    
    original_spacing= brain_mask_img.GetSpacing()
    
    d_img0= d_img.copy()
    d_img= np.zeros( (sx, sy, sz, 300) )
    for i in range(300):
        temp= d_img0[:,:,:,i]
        temp= np.transpose( temp, [2,1,0] )
        temp= sitk.GetImageFromArray(temp)
        temp.SetDirection( brain_mask_img.GetDirection() )
        temp.SetOrigin( brain_mask_img.GetOrigin() )
        temp.SetSpacing( brain_mask_img.GetSpacing() )
        temp= dk_seg.resample3d(temp, original_spacing, desired_spacing, sitk.sitkBSpline)
        temp= sitk.GetArrayFromImage(temp)
        temp= np.transpose( temp, [2,1,0] )
        d_img[:,:,:,i]= temp
    
    brain_mask_img= dk_seg.resample3d(brain_mask_img, original_spacing, desired_spacing, sitk.sitkNearestNeighbor)
    # sitk.WriteImage(brain_mask_img, dav_dir + 'brain_mask.nii.gz' )
    brain_mask= sitk.GetArrayFromImage(brain_mask_img)
    brain_mask= np.transpose( brain_mask, [2,1,0] )
    
    
    d_img_orig, b_vals_orig, b_vecs_orig= d_img.copy(), b_vals.copy(), b_vecs.copy()
    
    
    
    ref_dir= brain_mask_img.GetDirection()
    ref_org= brain_mask_img.GetOrigin()
    ref_spc= brain_mask_img.GetSpacing()
    
    skull= crl_aux.skull_from_brain_mask(brain_mask, radius= 2.0)
    brain_mask= np.logical_and(brain_mask==1, skull==0)
    
    b0_img= d_img[:,:,:,b_vals==0]
    b0_img= np.mean(b0_img, axis=-1)
    mask= b0_img>0
    
    gtab = gradient_table(b_vals, b_vecs)
    tenmodel = dti.TensorModel(gtab)
    
    tenfit = tenmodel.fit(d_img, brain_mask)
    
    # evecs = tenfit.evecs
    # evecs = evecs[..., 0]
    # evals = tenfit.evals
    # evals_1, evals_2, evals_3= evals[:,:,:,0], evals[:,:,:,1], evals[:,:,:,2]
    
    FA_full = fractional_anisotropy(tenfit.evals)
    FA_full[np.isnan(FA_full)] = 0
    # FA_full[skull==1]= 0
    
    temp= np.transpose(FA_full.astype(np.float32), [2,1,0]).astype(np.float32)
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'FA_full.nii.gz' )
    
    FA_img_nii= nib.load( results_dir + 'FA_full.nii.gz' )
    FA_img_nii.header['srow_x']= FA_img_nii.affine[0,:]
    FA_img_nii.header['srow_y']= FA_img_nii.affine[1,:]
    FA_img_nii.header['srow_z']= FA_img_nii.affine[2,:]
    affine= FA_img_nii.affine
            
            
    temp = mean_diffusivity(tenfit.evals)
    temp[np.isnan(temp)] = 0
    temp= np.transpose(temp.astype(np.float32), [2,1,0]).astype(np.float32)
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'MD_full.nii.gz' )
    # MD_full[skull==1]= 0
    
    CFA_img = color_fa(FA_full, tenfit.evecs)
    temp= np.transpose( CFA_img, [2,1,0,3] )
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'CFA_full.nii.gz')
    
    ten_full= tenfit.lower_triangular()
    temp= np.transpose( ten_full, [2,1,0,3] )
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'TEN_full.nii.gz')
    
    
    ten_odf_full= tenfit.odf(sphere_odf).astype(np.float32)
    
    '''qball_model = shm.QballModel(gtab, 8)
    peaks = dp.peaks_from_model(model=qball_model, data=d_img,
                relative_peak_threshold=.5,
                min_separation_angle=25,
                sphere=sphere_fod, mask=brain_mask)
    ap = shm.anisotropic_power(peaks.shm_coeff)
    nclass = 3
    beta = 0.1
    hmrf = TissueClassifierHMRF()
    initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass, beta)
    csf = PVE[..., 0]
    gm =  PVE[..., 1]
    wm =  PVE[..., 2]'''
    
    d_img=  d_img[:,:,:,b_vals==1000]
    b_vecs= b_vecs[:,b_vals==1000]
    b_vals= b_vals[b_vals==1000]
    
    '''grad_ind, grad_ang, _ = crl_aux.find_closest_bvecs_with_rotation(v_grad.T, b_vecs.T, n_rot= 90)
    
    if not len( np.unique(grad_ind) )==N_grad:
        print('                            Selected b_vecs are not unique!')
    else:
        print('mean and std of angles  %.3f ' % grad_ang.mean() , '  %.3f ' % grad_ang.std() ,  )
        
        d_img_0_1000=  np.concatenate( ( d_img_orig[:,:,:,b_vals_orig==0], d_img[:,:,:,grad_ind] ) , axis= -1)
        b_vecs_0_1000= np.concatenate( ( b_vecs_orig[:,b_vals_orig==0],    b_vecs[:,grad_ind] ) , axis= -1)
        b_vals_0_1000= np.concatenate( ( b_vals_orig[b_vals_orig==0],    b_vals[grad_ind] ) , axis= -1)
        gtab_0_1000 = gradient_table(b_vals_0_1000, b_vecs_0_1000)
        
        tenmodel = dti.TensorModel(gtab_0_1000)
        tenfit = tenmodel.fit(d_img_0_1000, brain_mask)
        FA_rot = fractional_anisotropy(tenfit.evals)
        FA_rot[np.isnan(FA_rot)] = 0
        FA_rot[skull==1]= 0
        CFA_rot= color_fa(FA_rot, tenfit.evecs)'''
    
    grad_ind_norot, grad_ang_norot, _ = crl_aux.find_closest_bvecs_no_rotation(v_grad.T, b_vecs.T, antipodal=True)
    # print('mean and std of angles  %.3f ' % grad_ang_norot.mean() , '  %.3f ' % grad_ang_norot.std() ,  )
    
    d_img_0_1000=  np.concatenate( ( d_img_orig[:,:,:,b_vals_orig==0], d_img[:,:,:,grad_ind_norot] ) , axis= -1)
    b_vecs_0_1000= np.concatenate( ( b_vecs_orig[:,b_vals_orig==0],    b_vecs[:,grad_ind_norot] ) , axis= -1)
    b_vals_0_1000= np.concatenate( ( b_vals_orig[b_vals_orig==0],      b_vals[grad_ind_norot] ) , axis= -1)
    gtab_0_1000 = gradient_table(b_vals_0_1000, b_vecs_0_1000)
    
    tenmodel = dti.TensorModel(gtab_0_1000)
    tenfit = tenmodel.fit(d_img_0_1000, brain_mask)
    
    FA_norot = fractional_anisotropy(tenfit.evals)
    FA_norot[np.isnan(FA_norot)] = 0
    temp= np.transpose(FA_norot.astype(np.float32), [2,1,0]).astype(np.float32)
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'FA_wlls.nii.gz' )
    
    MD_norot = mean_diffusivity(tenfit.evals)
    MD_norot[np.isnan(MD_norot)] = 0
    temp= np.transpose(MD_norot.astype(np.float32), [2,1,0]).astype(np.float32)
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'MD_wlls.nii.gz' )
    
    
    ten_norot= tenfit.lower_triangular()
    temp= np.transpose( ten_norot, [2,1,0,3] )
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'TEN_wlls.nii.gz')

    ten_odf_norot= tenfit.odf(sphere_odf).astype(np.float32)
    
    # FA_norot_err= np.abs( FA_norot-FA_full  )
    # MD_norot_err= np.abs( MD_norot-MD_full  )
    
    temp = color_fa(FA_norot, tenfit.evecs)
    temp= np.transpose( temp, [2,1,0,3] )
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'CFA_wlls.nii.gz')
    
    
    
    sig_img= d_img[:,:,:,grad_ind_norot]/np.tile( b0_img[:,:,:,np.newaxis], [1, 1, 1, N_grad])
    sig_img[sig_img>1]= 1
    sig_img[np.isnan(sig_img)]= 0
    sig_img[brain_mask==0]= 0
    
    ten_DL = np.zeros((SX, SY, SZ, n_class))
    # if LEARN_UNCERTAINTY:
    #     unc_DL = np.zeros((SX, SY, SZ, n_class))
    y_c = np.zeros((SX, SY, SZ))
    
    
    if MODEL== 'Transformer' or MODEL=='DTI_Transformer_plusplus':
        
        for lx in lx_list:
            for ly in ly_list:
                for lz in lz_list:
                    
                    if np.min(brain_mask[lx+LXc:lx+LX-LXc, ly+LYc:ly+LY-LYc, lz+LZc:lz+LZ-LZc]) > 0:
                        
                        batch_x = sig_img[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                        batch_x= np.reshape(batch_x, [1, L**3, n_channel])
                        
                        pred_temp = sess.run(Y_pred,  feed_dict={X: batch_x, p_keep:keep_test})
                        ten_DL[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[0, :, :, :, :]
                        
                        # if LEARN_UNCERTAINTY:
                        #     pred_temp = sess.run(U_pred,  feed_dict={X: batch_x, p_keep: keep_test})
                        #     unc_DL[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[0, :, :, :, :]
                            
                        y_c[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
   
    elif MODEL== 'CNN':
        
        for lx in lx_list:
            for ly in ly_list:
                for lz in lz_list:
                    
                    if np.max(brain_mask[lx+LXc:lx+LX-LXc, ly+LYc:ly+LY-LYc, lz+LZc:lz+LZ-LZc]) > 0:
                        
                        batch_x = sig_img[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :].copy()
                        batch_x= batch_x[np.newaxis,:]
                        
                        pred_temp = sess.run(Y_pred,  feed_dict={X: batch_x, p_keep_conv: keep_test})
                        ten_DL[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[0, :, :, :, :]
                        
                        # if LEARN_UNCERTAINTY:
                        #     pred_temp = sess.run(U_pred,  feed_dict={X: batch_x, p_keep_conv: keep_test})
                        #     unc_DL[lx:lx + LX, ly:ly + LY, lz:lz + LZ, :] += pred_temp[0, :, :, :, :]
                            
                        y_c[lx:lx + LX, ly:ly + LY, lz:lz + LZ] += 1
    
    ten_DL[brain_mask==0,:] = 0
    ten_DL= ten_DL/ np.tile( y_c[:,:,:,np.newaxis]+0.00001, [1, 1, 1, n_class])
    ten_DL/= 1000
    
    
    
    ten_odf_DL= np.zeros((SX, SY, SZ, len(v_odf)), np.float32)
    CFA_DL= np.zeros((SX, SY, SZ, 3))
    FA_DL= np.zeros((SX, SY, SZ))
    MD_DL= np.zeros((SX, SY, SZ))
    
    for ix in range(SX):
        for iy in range(SY):
            for iz in range(SZ):
                
                if brain_mask[ix, iy, iz]>0:
                    
                    wlls_params= ten_DL[ix,iy,iz,:]
                    temp= np.zeros(6)
                    temp[0]= wlls_params[0]
                    temp[1]= wlls_params[2]
                    temp[2]= wlls_params[5]
                    temp[3]= wlls_params[1]
                    temp[4]= wlls_params[3]
                    temp[5]= wlls_params[4]
                    
                    wlls_tensor= crl_dti.from_lower_triangular(temp)
                    
                    e_vals, e_vecs = crl_dti.evals_and_evecs_from_tensor(wlls_tensor)
                    
                    lower = 4 * np.pi * np.sqrt(np.prod(e_vals, -1))
                    projection = np.dot(v_odf, e_vecs)
                    projection /= np.sqrt(e_vals)
                    ten_odf_DL[ix, iy, iz,: ]= ((vector_norm(projection) ** -3) / lower).T
                    
                    FA_DL[ix,iy,iz], CFA_DL[ix,iy,iz] = crl_dti.cfa_from_tensor(wlls_tensor)
                    MD_DL[ix,iy,iz] = crl_dti.md_from_tensor(wlls_tensor)
                    
    ten_odf_DL[np.isnan(ten_odf_DL)]= 0
    
    temp= np.transpose(FA_DL.astype(np.float32), [2,1,0]).astype(np.float32)
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'FA_DL.nii.gz' )
    
    temp= np.transpose(MD_DL.astype(np.float32), [2,1,0]).astype(np.float32)
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'MD_DL.nii.gz' )
    
    temp= np.transpose( ten_DL, [2,1,0,3] )
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'TEN_DL.nii.gz')
    
    temp= np.transpose( CFA_DL, [2,1,0,3] )
    temp= sitk.GetImageFromArray(temp)
    temp.SetDirection(ref_dir)
    temp.SetOrigin(ref_org)
    temp.SetSpacing(ref_spc)
    sitk.WriteImage(temp, results_dir + 'CFA_DL.nii.gz')
    
    
    
    
    
    
    skull_seed= crl_aux.skull_from_brain_mask(brain_mask, radius= 4.0)
    brain_mask_seed= np.logical_and(brain_mask==1, skull_seed==0)
    
    
    
    odf_temp= ten_odf_full.copy()
    FA_array= FA_full.copy()
    
    FA_array[brain_mask_seed==0] = 0
    
    for sharpen in [1]:
        
        pmf = odf_temp.clip(min=0)
        pmf= pmf**sharpen
        
        fodf_pred_sum= np.sum(pmf, axis= -1)
        for i in range(pmf.shape[0]):
            for j in range(pmf.shape[1]):
                for k in range(pmf.shape[2]):
                    if fodf_pred_sum[i,j,k]>0:
                        pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
        
        for FA_temp in 10:
            
            if True: #tractography== 'whole-brain':
                
                seed_mask= FA_array>0.01*FA_temp
                #seed_mask= wm_img
                #seed_mask= ts_img==3
                #seed_mask= seed_mask *  (1-skull)
                seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
                
                stopping_criterion = ThresholdStoppingCriterion(FA_array, 0.01*FA_temp)
                
                peak_dg = ClosestPeakDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                              sphere=sphere_odf)
                streamline_generator = LocalTracking(peak_dg, stopping_criterion, seeds,
                                                          affine, step_size=.5)
                streamlines = Streamlines(streamline_generator)
                sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                save_trk(sft, results_dir + 'trk_baseline_EuDX_' +str(sharpen)+ '_' + str(FA_temp) +  '.trk', bbox_valid_check=False)
                
    odf_temp= ten_odf_DL.copy()
    FA_array= FA_full.copy()
    
    FA_array[brain_mask_seed==0] = 0
            
    for sharpen in [1]:
        
        pmf = odf_temp.clip(min=0)
        pmf= pmf**sharpen
        
        fodf_pred_sum= np.sum(pmf, axis= -1)
        for i in range(pmf.shape[0]):
            for j in range(pmf.shape[1]):
                for k in range(pmf.shape[2]):
                    if fodf_pred_sum[i,j,k]>0:
                        pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
        
        for FA_temp in [10]:
        
            if True: #results_dirtractography== 'whole-brain':
                
                seed_mask= FA_array>0.01*FA_temp
                #seed_mask= wm_img
                #seed_mask= ts_img==3
                #seed_mask= seed_mask *  (1-skull)
                seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
                
                stopping_criterion = ThresholdStoppingCriterion(FA_array, 0.01*FA_temp)
                
                peak_dg = ClosestPeakDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                              sphere=sphere_odf)
                streamline_generator = LocalTracking(peak_dg, stopping_criterion, seeds,
                                                          affine, step_size=.5)
                streamlines = Streamlines(streamline_generator)
                sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                save_trk(sft, results_dir + 'trk_dl_EuDX_' +str(sharpen)+ '_' + str(FA_temp) +  '.trk', bbox_valid_check=False)
                
    
    
    odf_temp= ten_odf_norot.copy()
    FA_array= FA_full.copy()
    
    
    FA_array[brain_mask_seed==0] = 0
    
    for sharpen in [1]:
        
        pmf = odf_temp.clip(min=0)
        pmf= pmf**sharpen
        
        fodf_pred_sum= np.sum(pmf, axis= -1)
        for i in range(pmf.shape[0]):
            for j in range(pmf.shape[1]):
                for k in range(pmf.shape[2]):
                    if fodf_pred_sum[i,j,k]>0:
                        pmf[i,j,k,:]/= fodf_pred_sum[i,j,k]
        
        for FA_temp in [10]:
        
            if True: #tractography== 'whole-brain':
                
                seed_mask= FA_array>0.01*FA_temp
                #seed_mask= wm_img
                #seed_mask= ts_img==3
                #seed_mask= seed_mask *  (1-skull)
                seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
                
                stopping_criterion = ThresholdStoppingCriterion(FA_array, 0.01*FA_temp)
                
                peak_dg = ClosestPeakDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                              sphere=sphere_odf)
                streamline_generator = LocalTracking(peak_dg, stopping_criterion, seeds,
                                                          affine, step_size=.5)
                streamlines = Streamlines(streamline_generator)
                sft = StatefulTractogram(streamlines, FA_img_nii, Space.RASMM)
                save_trk(sft, results_dir + 'trk_norot_EuDX_' +str(sharpen)+ '_' + str(FA_temp) +  '.trk', bbox_valid_check=False)
              








