from  visualize_fig import Visualize
vis = Visualize()

files = [((0, 400),"Privacy-utility tradeoff for arrival times", "result_scripts/loss_vs_privacy_occupancy_statistics_normal_deep_arrival.pickle"),
            ((0, 400), "Arrival times with public dataset", "result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_arrival.pickle"),
            ((0, 800), "Departure times with presanitized dataset", "result_scripts/loss_vs_privacy_occupancy_statistics_normal_deep_departure.pickle"),
            ((0, 800),"Privacy-utility tradeoff for departure times", "result_scripts/loss_vs_privacy_occupancy_statistics_public_deep_departure.pickle"),
            ((0, 3000),"Peak-time consumption with presanitized dataset", "result_scripts/loss_vs_privacy_energy_usage_normal_deep.pickle"),
            ((0, 3000),"Privacy-utility tradeoff for peak-time consumption", "result_scripts/loss_vs_privacy_energy_usage_public_deep.pickle"),
            ((0, 4),"Privacy-utility tradeoff for lunch time example", "result_scripts/loss_vs_privacy_occupancy_window_public_deep.pickle"),
            ((0, 4),"Lunch time with presanitized dataset", "result_scripts/loss_vs_privacy_occupancy_window_normal_deep.pickle"),]
for file in files:
    vis.infoloss_vs_level(file[0], file[1], file[2])