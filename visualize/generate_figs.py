from  visualize_fig import Visualize
vis = Visualize()

files = [("Publication specialized for arrival times with subsampled dataset", "result_scripts/loss_vs_privacy_occupancy_statistics_normal_deep_arrival.pickle"),
            ("Publication specialized for arrival times with public dataset", "result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_arrival.pickle"),
            ("Publication specialized for departure times with subsampled dataset", "result_scripts/loss_vs_privacy_occupancy_statistics_normal_deep_departure.pickle"),
            ("Publication specialized for departure times with public dataset", "result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_departure.pickle"),
            ("Publication specialized for peak energy usage with subsampled dataset", "result_scripts/loss_vs_privacy_energy_usage_normal_deep.pickle"),
            ("Publication specialized for peak energy usage with public dataset", "result_scripts/loss_vs_privacy_energy_usage_public_deep.pickle"),
            ("Publication specialized for occupancy pattern at Lunch with public dataset", "result_scripts/loss_vs_privacy_occupancy_window_public_deep.pickle"),
            ("Publication specialized for occupancy pattern at Lunch with subsample dataset", "result_scripts/loss_vs_privacy_occupancy_window_normal_deep.pickle"),]
for file in files:
    vis.infoloss_vs_level(file[0], file[1])