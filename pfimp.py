# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(imps,stds,title):
    idx = np.argsort(imps)[::-1] # ::-1 reveres the array
    cl = np.array(['indigo','blue','green','magenta','pink','yellow'])
    colors = cl[idx]
    ran = range(imps.shape[0])
    plt.title(title)
    plt.bar(ran,imps[idx],color=colors,yerr=stds[idx],align='center')
    # plt.bar(ran,imps[idx],color=colors,align='center')
    plt.grid(True)
    plt.xticks(ran,idx)
    plt.xlim([-1,imps.shape[0]])
    
def plot_summary_fimp(x):
    """ Helps in plotting summary of features"""
    n,d = x.shape
    ret = [np.zeros(d) for i in xrange(d)]
    xx = np.arange(d)
    for i in xrange(d):
        q,c = np.unique(x[:,i],return_counts=True)
        for iqq,qq in enumerate(q):
            ret[qq][i] = c[iqq]
    wut = np.array(ret) # 
    col = np.array(['indigo','blue','green','magenta','pink','yellow'])
    kt = np.zeros(d) # keep track
    for idx in xrange(d):
        if idx == 0:
            plt.bar(np.arange(d),wut[idx],color=col[idx],align='center',label=str(idx))
        else:
            plt.bar(np.arange(d),wut[idx],bottom=kt,color=col[idx],align='center',label=str(idx))
        kt += wut[idx]
