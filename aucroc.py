import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

regret_indist = np.load('./array/indist_regret.npy')
regret_ood = np.load('./array/ood_regret.npy')

combined = np.concatenate((regret_indist, regret_ood))
label_1 = np.ones(len(regret_indist))
label_2 = np.zeros(len(regret_ood))
label = np.concatenate((label_1, label_2))

fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)

#plot_roc_curve(fpr, tpr)

rocauc = metrics.auc(fpr, tpr)
print('AUC for likelihood regret is: ', rocauc)




nll_cifar = np.load('./array/indist_nll.npy')
nll_svhn = np.load('./array/ood_nll.npy')

combined = np.concatenate((nll_cifar, nll_svhn))
label_1 = np.ones(len(nll_cifar))
label_2 = np.zeros(len(nll_svhn))
label = np.concatenate((label_1, label_2))

fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)

#plot_roc_curve(fpr, tpr)

rocauc = metrics.auc(fpr, tpr)
print('AUC for nll is: ', rocauc)

