import pickle
import matplotlib.pyplot as plt
import numpy as np


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

plt.plot(anonymity_vec, losses_best, label='Ground truth metric')
plt.plot(anonymity_vec, losses_generic, label='Generic metric')
plt.plot(anonymity_vec, losses_linear, label='Linear metric')
plt.plot(anonymity_vec, losses_nonlinear,label='Nonlinear metric')
plt.legend()
plt.show()

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

plt.plot(anonymity_vec, losses_best, label='Ground truth metric')
plt.plot(anonymity_vec, losses_generic, label='Generic metric')
plt.plot(anonymity_vec, losses_linear, label='Linear metric')
plt.plot(anonymity_vec, losses_nonlinear,label='Nonlinear metric')
plt.legend()
plt.show()
