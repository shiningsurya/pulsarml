import numpy as np
import matplotlib.pyplot as plt

rem = 'ytest,ypredict,ypp0,ypp1'

def givemis(df,xl,xm,numPoints=100):
    # I'm gonna stay in 4-6 range
    tv = np.linspace(xl,xm,numPoints) # I guess 100 points would suffice
    mis1 = np.zeros((numPoints,)) # Misplaced pulsars
    mis0 = np.zeros((numPoints,)) # Misplaced RFI
    p1 = df[:,3] # The pulsar probabilities
    truth = df[:,0]
    # My loop starts here
    for idx,threshold in enumerate(tv):
        threshold_pulsars = p1 >= threshold
        # ^ Selecting pulsar probabilities which are greater than threshold
        threshold_pulsars = np.array(threshold_pulsars,dtype=np.int)
        thold = truth - threshold_pulsars # ytest - ypredict
        uq,counts = np.unique(thold,return_counts = True)
        # -1 -> RFI predicted as Pulsar  FP
        # 0 -> RFI predicted as RFI or Pulsar predicted as pulsar
        # 1 -> Pulsar weren't able to predict FN
        sc = np.sum(counts)
        for i,x in enumerate(uq):
            if x == -1:
                mis0[idx] = counts[i]/(sc*1.0)
            elif x == 1:
                mis1[idx] = counts[i]/(sc*1.0)
    return (tv,mis0,mis1)

def erts_plot(mis1,mis2,xlims):
    fig, ax = plt.subplots(1,2,sharey=True,figsize=(10,5))
    plt.sca(ax[0])
    plt.semilogy(mis1[0],mis1[1],label='Misplaced RFI',lw=2,c='b')
    plt.semilogy(mis1[0],mis1[2],label='Misplaced Pulsar',lw=2,c='g')
    zmis1 = (mis1[1] > 0) # last element
    zmis2 = (mis1[2] > 0) # first element
    plt.plot( [mis1[0][zmis1][-1] , mis1[0][zmis1][-1]] , [1e-6, mis1[1][zmis1][-1] ],c='b',lw=2)
    plt.plot( [mis1[0][zmis2][0]  , mis1[0][zmis2][0]]  , [1e-6, mis1[2][zmis2][0]  ],c='g',lw=2)
    plt.xlabel('Threshold Probabilities')
    plt.ylabel('Error Rate')
    plt.grid(True)
    plt.legend(loc='best')
    #
    plt.sca(ax[1])
    plt.semilogy(mis2[0],mis2[1],label='Misplaced RFI',lw=2,c='b')
    plt.semilogy(mis2[0],mis2[2],label='Misplaced Pulsar',lw=2,c='g')
    zmis21 = (mis2[1] > 0) # last element
    zmis22 = (mis2[2] > 0) # first element
    plt.plot( [mis2[0][zmis21][-1] , mis2[0][zmis21][-1]] , [1e-6, mis2[1][zmis21][-1] ],c='b',lw=2)
    plt.plot( [mis2[0][zmis22][0]  , mis2[0][zmis22][0]]  , [1e-6, mis2[2][zmis22][0]  ],c='g',lw=2)
    plt.xlim(xlims)
    # plt.xticks(np.arange(10),np.linspace(0.5,0.53,10),rotation=90)
    plt.xlabel('Threshold Probabilities $zoomed\ in$')
    # plt.ylabel('Error Rate')
    plt.grid(True)
    plt.legend(loc='best')

def areas(mis):
    mm = np.minimum(mis[1],mis[2])

    from scipy.integrate import simps

    sm0 = simps(mm,x=mis[0])
    s10 = simps(mis[1],x=mis[0])
    s20 = simps(mis[2],x=mis[0])

    print 'mmint',sm0
    print 'mis1',s10
    print 'mis2',s20
    # return (sm0,s10,s20)

def thres(tv,df):
    from sklearn.metrics import recall_score,precision_score
    p1 = df[:,3] # The pulsar probabilities
    truth = df[:,0]
    threshold_pulsars = p1 >= tv
    # ^ Selecting pulsar probabilities which are greater than threshold
    threshold_pulsars = np.array(threshold_pulsars,dtype=np.int)
    thold = truth - threshold_pulsars # ytest - ypredict
    uq,counts = np.unique(thold,return_counts = True)
    # -1 -> RFI predicted as Pulsar  FP
    # 0 -> RFI predicted as RFI or Pulsar predicted as pulsar
    # 1 -> Pulsar weren't able to predict FN
    sc = np.sum(counts)
    for i,x in enumerate(uq):
        if x == -1:
            print 'FPR:',counts[i]/(sc*1.0)
        elif x == 1:
            print 'FNR:',counts[i]/(sc*1.0)
    print 'Recall:',recall_score(truth,threshold_pulsars)
    print 'Precision:',precision_score(truth,threshold_pulsars)
