import pickle
import matplotlib.pyplot as plt
import numpy as np

# window
with open('result_scripts/loss_vs_privacy_occupancy_window_public_deep.pickle', 'rb') as f:
   _, losses, _ = pickle.load(f)

anonymity_vec = range(2, 8)
anonymity_n = len(anonymity_vec)

losses_linear = np.empty(anonymity_n)
losses_nonlinear = np.empty(anonymity_n)
losses_best = np.empty(anonymity_n)
losses_generic = np.empty(anonymity_n)

for i in range(anonymity_n):
    losses_best[i] = losses[anonymity_vec[i]][0]
    losses_generic[i] = losses[anonymity_vec[i]][1]
    losses_linear[i] = losses[anonymity_vec[i]][2]
    losses_nonlinear[i] = losses[anonymity_vec[i]][3]

plt.plot(anonymity_vec, losses_best,'o--', label='Ground truth metric')
plt.plot(anonymity_vec, losses_generic, 's--',label='Generic metric')
plt.plot(anonymity_vec, losses_linear, '^--',label='Linear metric')
plt.plot(anonymity_vec, losses_nonlinear,'X--',label='Nonlinear metric')
plt.xlabel('Anonymity level')
plt.ylabel('Information loss')
plt.title('Privacy-utility tradeoff for lunchtime example')
plt.legend()
plt.show()
plt.savefig("visualize/figures/tradeoff_lunch_notall", bbox_inches='tight',dpi=100)


# energy usage
with open('result_scripts/loss_vs_privacy_energy_usage_public_deep.pickle', 'rb') as f:
   _, losses, _ = pickle.load(f)

anonymity_vec = range(2, 8)
anonymity_n = len(anonymity_vec)

losses_linear = np.empty(anonymity_n)
losses_nonlinear = np.empty(anonymity_n)
losses_best = np.empty(anonymity_n)
losses_generic = np.empty(anonymity_n)

for i in range(anonymity_n):
    losses_best[i] = losses[anonymity_vec[i]][0]
    losses_generic[i] = losses[anonymity_vec[i]][1]
    losses_linear[i] = losses[anonymity_vec[i]][2]
    losses_nonlinear[i] = losses[anonymity_vec[i]][3]

plt.plot(anonymity_vec, losses_best,'o--', label='Ground truth metric')
plt.plot(anonymity_vec, losses_generic, 's--',label='Generic metric')
plt.plot(anonymity_vec, losses_linear, '^--',label='Linear metric')
plt.plot(anonymity_vec, losses_nonlinear,'X--',label='Nonlinear metric')
plt.xlabel('Anonymity level')
plt.ylabel('Information loss (W)')
plt.title('Privacy-utility tradeoff for peak-time consumption')
plt.legend()
plt.show()
plt.savefig("visualize/figures/tradeoff_energy_notall", bbox_inches='tight',dpi=100)



# departure
with open('result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_departure_all.pickle', 'rb') as f:
   _, losses, _ = pickle.load(f)

anonymity_vec = range(2, 8)
anonymity_n = len(anonymity_vec)

losses_linear = np.empty(anonymity_n)
losses_nonlinear = np.empty(anonymity_n)
losses_best = np.empty(anonymity_n)
losses_generic = np.empty(anonymity_n)

for i in range(anonymity_n):
    losses_best[i] = losses[anonymity_vec[i]][0]
    losses_generic[i] = losses[anonymity_vec[i]][1]
    losses_linear[i] = losses[anonymity_vec[i]][2]
    losses_nonlinear[i] = losses[anonymity_vec[i]][3]

plt.plot(anonymity_vec, losses_best,'o--', label='Ground truth metric')
plt.plot(anonymity_vec, losses_generic, 's--',label='Generic metric')
plt.plot(anonymity_vec, losses_linear, '^--',label='Linear metric')
plt.plot(anonymity_vec, losses_nonlinear,'X--',label='Nonlinear metric')
plt.xlabel('Anonymity level')
plt.ylabel('Information loss')
plt.title('Privacy-utility tradeoff for departure time')
plt.legend()
plt.show()
plt.savefig("visualize/figures/tradeoff_departure_all", bbox_inches='tight',dpi=100)


# arrival
with open('result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_arrival_all.pickle', 'rb') as f:
   _, losses, _ = pickle.load(f)

anonymity_vec = range(2, 8)
anonymity_n = len(anonymity_vec)

losses_linear = np.empty(anonymity_n)
losses_nonlinear = np.empty(anonymity_n)
losses_best = np.empty(anonymity_n)
losses_generic = np.empty(anonymity_n)

for i in range(anonymity_n):
    losses_best[i] = losses[anonymity_vec[i]][0]
    losses_generic[i] = losses[anonymity_vec[i]][1]
    losses_linear[i] = losses[anonymity_vec[i]][2]
    losses_nonlinear[i] = losses[anonymity_vec[i]][3]

plt.plot(anonymity_vec, losses_best,'o--', label='Ground truth metric')
plt.plot(anonymity_vec, losses_generic, 's--',label='Generic metric')
plt.plot(anonymity_vec, losses_linear, '^--',label='Linear metric')
plt.plot(anonymity_vec, losses_nonlinear,'X--',label='Nonlinear metric')
plt.xlabel('Anonymity level')
plt.ylabel('Information loss')
plt.title('Privacy-utility tradeoff for arrival time')
plt.legend()
plt.show()
plt.savefig("visualize/figures/tradeoff_arrival_all", bbox_inches='tight',dpi=100)