import pickle
import matplotlib.pyplot as plt
import numpy as np






## arrival
with open('result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_arrival_mc.pickle', 'rb') as f:
   _, losses, _, losses_best, losses_generic = pickle.load(f)

anonymity_vec = range(2, 8)
anonymity_n = len(anonymity_vec)
mc_num = 5

losses_linear = np.empty((anonymity_n,mc_num))
losses_nonlinear = np.empty((anonymity_n,mc_num))

for i in range(anonymity_n):
   for j in range(mc_num):
      losses_linear[i,j] = losses[(anonymity_vec[i],j)][2]
      losses_nonlinear[i,j] = losses[(anonymity_vec[i],j)][3]


plt.plot(anonymity_vec,losses_best,'X-',label='Ground truth metric')
plt.plot(anonymity_vec,losses_generic, 's-',label='Generic metric')
plt.errorbar(anonymity_vec,np.mean(losses_linear,axis=1),np.std(losses_linear,axis=1),label='Linear metric',fmt='--o')
plt.errorbar(anonymity_vec,np.mean(losses_nonlinear,axis=1),np.std(losses_nonlinear,axis=1),label='Nonlinear metric',fmt='-.v')
plt.legend()
plt.xlabel('Anonymity level')
plt.ylabel('Information loss (W)')
plt.xticks(anonymity_vec)
plt.title('Privacy-utility tradeoff for arrival time')
plt.show()
plt.savefig("visualize/figures/tradeoff_arrival_mc", bbox_inches='tight',dpi=100)



## departure

with open('result_scripts/loss_vs_privacy_occupancy_departure_public_deep_departure_mc.pickle', 'rb') as f:
   _, losses, _, losses_best, losses_generic = pickle.load(f)

anonymity_vec = range(2, 8)
anonymity_n = len(anonymity_vec)
mc_num = 5

losses_linear = np.empty((anonymity_n,mc_num))
losses_nonlinear = np.empty((anonymity_n,mc_num))

for i in range(anonymity_n):
   for j in range(mc_num):
      losses_linear[i,j] = losses[(anonymity_vec[i],j)][2]
      losses_nonlinear[i,j] = losses[(anonymity_vec[i],j)][3]


plt.plot(anonymity_vec,losses_best,'X-',label='Ground truth metric')
plt.plot(anonymity_vec,losses_generic, 's-',label='Generic metric')
plt.errorbar(anonymity_vec,np.mean(losses_linear,axis=1),np.std(losses_linear,axis=1),label='Linear metric',fmt='--o')
plt.errorbar(anonymity_vec,np.mean(losses_nonlinear,axis=1),np.std(losses_nonlinear,axis=1),label='Nonlinear metric',fmt='-.v')
plt.legend()
plt.xlabel('Anonymity level')
plt.ylabel('Information loss (W)')
plt.xticks(anonymity_vec)
plt.title('Privacy-utility tradeoff for departure time')
plt.show()
plt.savefig("visualize/figures/tradeoff_departure_mc", bbox_inches='tight',dpi=100)


# window
with open('result_scripts/loss_vs_privacy_occupancy_window_public_deep_mc.pickle', 'rb') as f:
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


plt.plot(anonymity_vec,losses_best,'X-',label='Ground truth metric')
plt.plot(anonymity_vec,losses_generic, 's-',label='Generic metric')
plt.errorbar(anonymity_vec,np.mean(losses_linear,axis=1),np.std(losses_linear,axis=1),label='Linear metric',fmt='--o')
plt.errorbar(anonymity_vec,np.mean(losses_nonlinear,axis=1),np.std(losses_nonlinear,axis=1),label='Nonlinear metric',fmt='-.v')
plt.legend()
plt.xlabel('Anonymity level')
plt.ylabel('Information loss (W)')
plt.xticks(anonymity_vec)
plt.title('Privacy-utility tradeoff for lunchtime example')
plt.show()
plt.savefig("visualize/figures/tradeoff_lunchtime_mc", bbox_inches='tight',dpi=100)



# energy
with open('result_scripts/loss_vs_privacy_energy_usage_public_deep_mc_high.pickle', 'rb') as f:
   _, losses, _, losses_best, losses_generic = pickle.load(f)

anonymity_vec = range(2,8)
anonymity_n = len(anonymity_vec)
mc_num = 5

losses_linear = np.empty((anonymity_n,mc_num))
losses_nonlinear = np.empty((anonymity_n,mc_num))

for i in range(anonymity_n):
   for j in range(mc_num):

      losses_linear[i,j] = losses[(anonymity_vec[i],j)][2]
      losses_nonlinear[i,j] = losses[(anonymity_vec[i],j)][3]


plt.plot(anonymity_vec,losses_best,'X-',label='Ground truth metric')
plt.plot(anonymity_vec,losses_generic, 's-',label='Generic metric')
plt.errorbar(anonymity_vec,np.mean(losses_linear,axis=1),np.std(losses_linear,axis=1),label='Linear metric',fmt='--o')
plt.errorbar(anonymity_vec,np.mean(losses_nonlinear,axis=1),np.std(losses_nonlinear,axis=1),label='Nonlinear metric',fmt='-.v')
plt.legend()
plt.xlabel('Anonymity level')
plt.ylabel('Information loss (W)')
plt.xticks(anonymity_vec)
plt.title('Privacy-utility tradeoff for peak-time consumption')
plt.show()
plt.savefig("visualize/figures/tradeoff_energy_mc", bbox_inches='tight',dpi=100)

