from  visualize_fig import Visualize
vis = Visualize()

files = [("Publication specialized for arrival times with subsampled dataset", "result_scripts/loss_vs_privacy_occupancy_statistics_normal_deep_arrival.pickle"),
            ("Publication specialized for arrival times with public dataset", "result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_arrival.pickle"),
            ("Publication specialized for departure times with subsampled dataset", "result_scripts/loss_vs_privacy_occupancy_statistics_normal_deep_departure.pickle"),
            ("Publication specialized for departure times with public dataset", "result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_departure.pickle")
            ]
for file in files:
    vis.infoloss_vs_level(file[0], file[1])