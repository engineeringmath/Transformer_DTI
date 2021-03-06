# -*- coding: utf-8 -*-
"""

Models for segmentation

@author: davoo
"""


import numpy as np
import tensorflow as tf




def davood_net(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001):
    
    feat_fine = [None] * (depth - 1)
        
    for level in range(depth):
        
        ks = ks_0
                
        if level == 0:
            
            strd = 1
            
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
        else:
            
            strd = 2
            n_l = n_channel * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_init'
            name_b = 'b_' + str(level) + '_init'
            W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_channel, n_feat_0], stddev=s_dev), name=name_w)
            b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1, level):
                n_l = n_feat_0 * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                name_w = 'W_' + str(level) + '_' + str(i) + '_init'
                name_b = 'b_' + str(level) + '_' + str(i) + '_init'
                W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat_0, n_feat_0], stddev=s_dev), name=name_w)
                b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                inp= tf.nn.dropout(inp, p_keep_conv)
                
            for level_reg in range(0, level):
                
                inp_0 = feat_fine[level_reg]
                
                level_diff = level - level_reg
                
                n_feat = n_feat_0 * 2 ** level_reg
                n_l = n_feat * ks ** 3
                s_dev = np.sqrt(2.0 / n_l)
                
                for j in range(level_diff):
                    name_w = 'W_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_b = 'b_' + str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
                    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tf.nn.relu(
                        tf.add(tf.nn.conv3d(inp_0, W_1, strides=[1, strd, strd, strd, 1], padding='SAME'), b_1))
                    inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp = tf.concat([inp, inp_0], 4)
                
        ks = ks_0
        
        n_feat = n_feat_0 * 2 ** level
        
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_down'
            name_b = 'b_' + str(level) + '_2_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_down'
            name_b = 'b_' + str(level) + '_3_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_down'
            name_b = 'b_' + str(level) + '_4_down'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_down'
            name_b = 'b_' + str(level) + '_5_down'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
            
            inp = inp + inp_1 + inp_0  ###
            
        if level < depth - 1:
            feat_fine[level] = inp
            
    # DeConvolution Layers
    
    for level in range(depth - 2, -1, -1):
        
        ks = ks_0
        
        n_l = n_feat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_up'
        name_b = 'b_' + str(level) + '_up'
        W_deconv = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat // 2, n_feat], stddev=s_dev), name=name_w)
        b_deconv = tf.Variable(tf.constant(bias_init, shape=[n_feat // 2]), name=name_b)
        in_shape = tf.shape(inp)
        
        out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        
        # if level == 3:
        #     out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, 9, in_shape[4] // 2])
        # else:
        #     out_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, in_shape[4] // 2])
        
        Deconv = tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
        Deconv = tf.nn.relu(tf.add(Deconv, b_deconv))
        Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp = tf.concat([feat_fine[level], Deconv], 4)
           
        if level == depth - 2:
            n_concat = n_feat
        else:
            n_concat = n_feat * 3 // 4
            
        if level < depth - 2:
            n_feat = n_feat // 2
            
        n_l = n_concat * ks ** 3
        s_dev = np.sqrt(2.0 / n_l)
        name_w = 'W_' + str(level) + '_1_up'
        name_b = 'b_' + str(level) + '_1_up'
        W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_concat, n_feat], stddev=s_dev), name=name_w)
        b_1 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1))
        inp= tf.nn.dropout(inp, p_keep_conv)
           
        if level > -1:
            
            inp_0 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_2_up'
            name_b = 'b_' + str(level) + '_2_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
               
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_3_up'
            name_b = 'b_' + str(level) + '_3_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
               
            inp = inp + inp_0  ###
            
        if level > -1:
            
            inp_1 = inp  ###
            
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_4_up'
            name_b = 'b_' + str(level) + '_4_up'
            W_2 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_2 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            inp= tf.nn.dropout(inp, p_keep_conv)
               
            n_l = n_feat * ks ** 3
            s_dev = np.sqrt(2.0 / n_l)
            name_w = 'W_' + str(level) + '_5_up'
            name_b = 'b_' + str(level) + '_5_up'
            W_3 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_feat], stddev=s_dev), name=name_w)
            b_3 = tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tf.nn.relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            inp= tf.nn.dropout(inp, p_keep_conv)
               
            inp = inp + inp_1 + inp_0  ###
            
     #        if level==0:
     #            n_l= n_feat*ks**3
     #            s_dev= np.sqrt(2.0/n_l)
     #            name_w= 'W_up'
     #            name_b= 'b_up'
     #            name_c= 'Conv_up'
     #            W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_class,n_feat], stddev=s_dev), name=name_w)
     #            b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
     #            in_shape = tf.shape(inp)
     #            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, n_class])
     #            Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
     #            output= tf.add(Deconv, b_deconv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w = 'W_out'
    name_b = 'b_out'
#    name_c = 'Conv_out'
    W_1 = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, n_class], stddev=s_dev), name=name_w)
    b_1 = tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
    output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    n_l = n_feat * ks ** 3
    s_dev = np.sqrt(2.0 / n_l)
    name_w_s = 'W_out_s'
    name_b_s = 'b_out_s'
#    name_c_s = 'Conv_out_s'
    W_1_s = tf.Variable(tf.truncated_normal([ks, ks, ks, n_feat, 1], stddev=s_dev), name=name_w_s)
    b_1_s = tf.Variable(tf.constant(bias_init, shape=[1]), name=name_b_s)
    output_s = tf.add(tf.nn.conv3d(inp, W_1_s, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1_s)
    # output= tf.nn.dropout(output, p_keep_conv)
    
    '''if write_summaries:
        tf.summary.histogram(name_w, W_1)
        tf.summary.histogram(name_b, b_1)
        tf.summary.image(name_c, slice_and_scale(inp))'''
        
    return output, output_s












def DTI_Transformer(X, T, pos_emb, n_inp, d_emb, ViT_depth, n_head, d_emb_h, n_channel, L, n_class=6, use_token=False, 
            residuals=True, layer_norm=True, learn_embd=False, p_keep=1.0, w_std= 0.002):
    
    A_list = [None] * (ViT_depth*n_head)
    
    inp = X
    
    n_feat_in=  n_channel
    n_feat_out= d_emb
    
    W_fc = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_fc')
    b_fc = tf.Variable(tf.zeros([n_feat_out]), name='b_fc')
    
    inp = tf.nn.relu( tf.matmul(inp, W_fc) + b_fc )
    inp = tf.nn.dropout(inp, p_keep)
    
    if use_token:
        inp= tf.concat((T, inp), axis=1)
    
    if learn_embd:
        
        # if use_token:
        #     pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb+1], stddev= 0.1), name='pos_emb')
        # else:
        #     pos_emb = tf.Variable(tf.truncated_normal([n_patch, d_emb], stddev= 0.1), name='pos_emb')
        
        inp= inp + pos_emb
    
    
    
    for i_depth in range(ViT_depth):
        
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb_h
        
        for i_head in range(n_head):
            
            W_q = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_q_' + str(i_depth) + '_' + str(i_head) )
            Q =   tf.matmul(inp, W_q)
            Q = tf.nn.dropout(Q, p_keep)
            
            W_k = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_k_' + str(i_depth) + '_' + str(i_head) )
            K =   tf.matmul(inp, W_k)
            K = tf.nn.dropout(K, p_keep)
            
            W_v = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_v_' + str(i_depth) + '_' + str(i_head) )
            V =   tf.matmul(inp, W_v)
            V = tf.nn.dropout(V, p_keep)
            
            A= tf.nn.softmax( tf.matmul(Q, K, transpose_b=True) / d_emb_h**0.5, axis= -1, name='A_' + str(i_depth) + '_' + str(i_head))
            # A= tf.matmul(Q, K, transpose_b=True)
            A_list[i_depth*n_head+i_head]= A
            
            SA= tf.matmul(A, V)
            
            if i_head==0:
                
                new_inp= SA
            
            else:
                
                new_inp= tf.concat((new_inp, SA), axis=-1)
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                    
        if layer_norm:
            inp= tf.contrib.layers.layer_norm(inp)
        
        n_feat_in=  d_emb
        n_feat_out= d_emb
        
        W_l = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(w_std/n_feat_out)), name='W_L_' + str(i_depth) )
        b_l = tf.Variable(tf.zeros([n_feat_out]), name='b_L_' + str(i_depth) )
        
        new_inp = tf.nn.relu( tf.matmul(inp, W_l) + b_l )
        new_inp = tf.nn.dropout(new_inp, p_keep)
        
        if residuals:
            inp= inp + new_inp
        else:
            inp= new_inp
                
        
    if use_token:
        
        out = inp[:,0,:]
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_L_out' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_L_out' )
        
        out = tf.matmul(out, W_oout) + b_out
    
    else:
        
        n_feat_in=  d_emb
        n_feat_out= n_class
        
        W_oout = tf.Variable(tf.truncated_normal([n_feat_in, n_feat_out], stddev= np.sqrt(2.0/n_feat_out)), name='W_O_0' )
        b_out = tf.Variable(tf.zeros([n_feat_out]), name='b_O_0' )
        
        # out = tf.nn.relu( tf.matmul(inp, W_oout) + b_out )
        out = tf.matmul(inp, W_oout) + b_out
        
        out = tf.reshape(out, tf.stack((-1, L, L, L, n_class)))
        
    
    return out, A_list













        