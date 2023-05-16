import numpy as np
import matplotlib.pyplot as plt

l = np.array([0.6840,0.7077,0.7038,0.7068])
#l = np.array([0.7154, 0.7170, 0.7193, 0.7177])
a = np.array([0.25, 0.50,0.75,1.0])
#a = np.array([0, 0.25,0.50,0.75])

plt.plot(a,l)
plt.ylabel('ROC-AUC on PPI')
plt.xlabel('Gradient weight a')
plt.savefig('/export/data/rli/Project/new/Simg/unsupervised_TU/collapse/' + 'tranfer_weight' + '.png')