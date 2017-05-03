import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.winter,colmap=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    k = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if colmap:
        plt.colorbar()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print("Normalized confusion matrix")
    # else:
    #print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
    return k

def cfm3in1(cfs,classes,cmap=plt.cm.RdPu):
    """
    To make it beautiful.
    Really beautiful.
    """
    if len(cfs) != 3:
        print 'Oh boi'
    tick_marks = np.arange(len(classes))
    fig , ax = plt.subplots(1,3,figsize=(10,5))
    #fig.suptitle('Adaboost')
    ###
    plt.sca(ax[0]) # left most
    k = plot_confusion_matrix(cfs[0],classes,title='$1^{st}$ iteration',cmap=cmap)
    plt.ylabel('True label')
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    ###
    plt.sca(ax[1])
    k = plot_confusion_matrix(cfs[1],classes,title='$2^{nd}$ iteration',cmap=cmap)
    plt.xlabel('Predicted label')
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks,[])
    ###
    plt.sca(ax[2])
    k = plot_confusion_matrix(cfs[2],classes,title='$3^{rd}$ iteration',cmap=cmap)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks,[])
    ###
    cbar_ax = fig.add_axes([0.13,0.08,0.75,0.05])
    fig.colorbar(k,cax=cbar_ax,orientation='horizontal')
    # fig.tight_layout()

def plot_feature_importance(imps,title):
    idx = np.argsort(imps)[::-1] # ::-1 reveres the array
    cl = np.array(['indigo','blue','green','magenta','pink','yellow'])
    colors = cl[idx]
    ran = range(imps.shape[0])
    plt.title(title)
    #plt.bar(ran,imps[idx],color='g',yerr=stds[idx],align='center')
    plt.bar(ran,imps[idx],color=colors,align='center')
    plt.grid(True)
    plt.xticks(ran,idx)
    plt.xlim([-1,imps.shape[0]])

def fimp3in1(fimpl):
    a = ['st','nd','rd']
    fig, ax = plt.subplots(1,3,sharey=True,figsize=(10,5))
    for i in xrange(3):
        plt.sca(ax[i])
        fx = fimpl[i]/(1.*fimpl[i].max())
        plot_feature_importance(fx,'$%s ^ {%s}$ iteration'%(str(i+1),a[i]))
