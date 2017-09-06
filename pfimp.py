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
