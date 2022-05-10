import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('logs/predictions_default.h5','r')

pdgs = f['labels'][:]
probs = f['probs'][:,0]

labs = np.zeros_like(pdgs)
labs[pdgs==-12] = 0
labs[pdgs== 12] = 1
labs[pdgs==-14] = 2
labs[pdgs== 14] = 3

####################################################
# Sample composition
plt.hist(labs, bins=4, range=(0,4), histtype='step')

plt.gca().set_xticks([0.5,1.5,2.5,3.5,4.5])
plt.gca().set_xticklabels(['NueBar','Nue','NumuBar','Numu'], ha='center')

plt.savefig('plots/sample.pdf')
plt.close() 

####################################################
# plt scores for all nus and anti-nus
plt.hist([ probs[pdgs>0],probs[pdgs<0] ], bins=100, range=(0,1), 
         histtype='step', color=['blue', 'red'], label=['nu','anti-nu'])

plt.legend(loc='upper center')
plt.xlabel('+ score')

plt.savefig('plots/fullscores.png')
plt.close()

####################################################
# plt scores for all nues
plt.hist([ probs[pdgs==12],probs[pdgs==-12] ], bins=100, range=(0,1), 
         histtype='step', color=['blue', 'red'], label=['nue','nuebar'])

plt.legend(loc='upper center')
plt.xlabel('+ score')

plt.savefig('plots/nuescores.png')
plt.close()

####################################################
# plt scores for all numus
plt.hist([ probs[pdgs==14],probs[pdgs==-14] ], bins=100, range=(0,1), 
         histtype='step', color=['blue', 'red'], label=['numu','numubar'])

plt.legend(loc='upper center')
plt.xlabel('+ score')

plt.savefig('plots/numuscores.png')
plt.close()

