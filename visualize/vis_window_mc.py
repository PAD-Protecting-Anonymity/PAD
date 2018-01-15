import pickle
import matplotlib.pyplot as plt

with open('result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_arrival_mc.pickle', 'rb') as f:
   _, losses, _, losses_best, losses_generic = pickle.load(f)

anonymity_vec = range(2, 8)


plt.plot(anonymity_vec,losses_best)
plt.plot(anonymity_vec,losses_generic)
plt.errorbar(anonymity_vec,)

