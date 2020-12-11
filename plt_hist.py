import numpy as np
import matplotlib.pyplot as plt


ll_indist = -np.load('./array/indist_nll.npy')
ll_ood = -np.load('./array/ood_nll.npy')


min_ll = min(ll_indist.min(), ll_ood.min())
max_ll = max(ll_indist.max(), ll_ood.max())
bins_ll = np.linspace(min_ll, max_ll, 50)

plt.hist(ll_indist, bins_ll, alpha=0.5, label='In-distribution')
plt.hist(ll_ood, bins_ll, alpha=0.5, label='OOD')

plt.legend(loc='upper right')
plt.show()

##############################################################################################
regret_indist = np.load('./array/indist_regret.npy')
regret_ood = np.load('./array/ood_regret.npy')

min_regret = min(regret_indist.min(), regret_ood.min())
max_regret = max(regret_indist.max(), regret_ood.max())
bins_regret = np.linspace(min_regret, max_regret, 50)

plt.hist(regret_indist, bins_regret, alpha=0.5, label='In-distribution')
plt.hist(regret_ood, bins_regret, alpha=0.5, label='OOD')

plt.legend(loc='upper right')
plt.show()
