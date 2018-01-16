import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('result_scripts/loss_vs_privacy_energy_usage_public_deep_mc.pickle', 'rb') as f:
   _, losses, _, losses_best, losses_generic = pickle.load(f)

anonymity_vec = range(2, 7)
anonymity_n = len(anonymity_vec)
mc_num = 5

losses_linear = np.empty((anonymity_n,mc_num))
losses_nonlinear = np.empty((anonymity_n,mc_num))

for i in range(anonymity_n):
   for j in range(mc_num):
      losses_linear[i,j] = losses[(anonymity_vec[i],j)][2]
      losses_nonlinear[i,j] = losses[(anonymity_vec[i],j)][3]


plt.plot(anonymity_vec,losses_best,label='Ground truth metric')
plt.plot(anonymity_vec,losses_generic,label='Generic metric')
plt.errorbar(anonymity_vec,np.mean(losses_linear,axis=1),np.std(losses_linear,axis=1),label='Linear metric')
plt.errorbar(anonymity_vec,np.mean(losses_nonlinear,axis=1),np.std(losses_nonlinear,axis=1),label='Nonlinear metric')
plt.legend()
plt.show()

