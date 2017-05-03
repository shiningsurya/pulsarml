# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(imps,stds,title):
     idx = np.argsort(imps)[::-1] # ::-1 reveres the array
     ran = range(imps.shape[0])
     plt.title(title+'.Feature Importance')
     #plt.bar(ran,imps[idx],color='g',yerr=stds[idx],align='center')
     plt.bar(ran,imps[idx],color='g',align='center')
     plt.grid(True)
     plt.xticks(ran,idx)
     plt.xlim([-1,imps.shape[0]])
