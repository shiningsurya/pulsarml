# function definition for ROC and AuROC
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc

def plot_roc(hmega,hprob,hsprob):
    """
    Plots ROC with some cool visualizations.
    """
    tpr_list  = []
    auc_list  = []
    stpr_list = []
    sauc_list = []
    mean_fpr  = np.linspace(0,1,100)

    for i in xrange(100):
        fp,tp,_ = roc_curve(hmega[i],hprob[i,:,1])
        sfp, stp= roc_curve(hmega[i],hsprob[i,:,1])
        tpr_list.append(interp(mean_fpr,fp,tp))
        stpr_list.append(interp(mean_fpr,sfp,stp))
        tpr_list[-1][0] = stpr_list[-1][0] = 0.0
        auc_list.append(auc(fp,tp))
        sauc_list.append(acu(sfp,stp))

    mean_tpr = np.mean(tpr_list,0)
    mean_stpr = np.mean(stpr_list,0)
    mean_tpr[-1] = mean_stpr[-1] =  1.0
    mean_auc = auc(mean_fpr,mean_tpr)
    mean_sauc = auc(mean_fpr,mean_stpr)
    std_auc = np.std(auc_list)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
    std_tpr = np.std(tpr_list,0)
    tp_up = np.minimum(mean_tpr + std_tpr, 1)
    tp_lw = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tp_lw, tp_up, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Random', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")

def cal_auroc(hmega,hprob,hsprob):
    """
    Calculates AuROC
    """
    ll = hmega.shape[0]
    tpr_list  = []
    auc_list  = []
    stpr_list = []
    sauc_list = []
    mean_fpr  = np.linspace(0,1,100)

    for i in xrange(ll):
        fp,tp,_ = roc_curve(hmega[i],hprob[i,:,1])
        sfp,stp,_= roc_curve(hmega[i],hsprob[i,:,1])
        tpr_list.append(interp(mean_fpr,fp,tp))
        stpr_list.append(interp(mean_fpr,sfp,stp))
        tpr_list[-1][0] = stpr_list[-1][0] = 0.0
        auc_list.append(auc(fp,tp))
        sauc_list.append(auc(sfp,stp))

    mean_tpr = np.mean(tpr_list,0)
    mean_stpr = np.mean(stpr_list,0)
    mean_tpr[-1] = mean_stpr[-1] =  1.0
    mean_auc = auc(mean_fpr,mean_tpr)
    mean_sauc = auc(mean_fpr,mean_stpr)
    std_auc = np.std(auc_list)
    std_sauc = np.std(sauc_list)
    ret = dict()
    ret["non_smote_tpr"] = mean_tpr
    ret["smote_tpr"] = mean_stpr
    ret["non_smote_auc"] = (mean_auc,std_auc)
    ret["smote_auc"] = (mean_sauc,std_sauc)
    ret["mean_fpr"] = mean_fpr
    return ret
