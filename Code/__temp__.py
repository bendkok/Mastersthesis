# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:06:54 2022

@author: benda
"""

import tensorflow as tf
import sys
import numpy as np





sess = tf.compat.v1.Session()
with sess.as_default():
    
    a = tf.math.log(0.001 * 0.0001 * 0)

    print(a.numpy())
    
    b = tf.math.exp(-0.082 * 1000.)

    print(b.numpy())
    
    c = tf.math.divide(1., 1e40)

    print(c.numpy())
    
#     V = -84.624
#     h = 0.988
    
#     alpha_h = 5.497962438709065e-10 * tf.exp(-0.25 * V)
#     beta_h = 1.7 / (1 + 0.1580253208896478 * tf.exp(-0.082 * V))
#     res = (1. - h) * alpha_h - beta_h * h
    
#     # with tf.Session() as sess:
#     #     print(alpha_h, beta_h, res)
    
#     print(alpha_h.eval(),  beta_h.eval(), res.eval())
    
#     g_s=0.0001
    
#     y = [0.011, 0.988, 0.975, 0.0001, 0.003, 0.994, 0.0001, -84.624]
    
#     E_s = -82.3 - 13.0287 * tf.math.log(0.001 * y[3])
#     i_s = g_s * (-E_s + V) * y[4] * y[5]
#     res0 = 7.000000000000001e-06 - 0.07 * y[3] - 0.01 * i_s
    
#     print(E_s.eval(),  i_s.eval(), res0.eval())
    
# print(type(float('nan')))
