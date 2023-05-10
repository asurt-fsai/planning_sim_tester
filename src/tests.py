import py_planning_trees
from planning_sim_tester import SimConfig
from planning_sim_tester import Simulator
from planning_sim_tester import Track, TrackConfig
import numpy as np
import matplotlib.pyplot as plt
import time

final_cones = None


class PlannerWrapper:
    def __init__(self):
        self.counter = 0

    def getPath(self, cones, verbose=False):
        global final_cones
        final_cones = cones
        conesX = cones[:, 0].tolist()
        conesY = cones[:, 1].tolist()
        conesType = cones[:, 2].astype(int).tolist()
        tic = time.time()
        (
            path,
            all_paths,
            waypoints,
            costs,
            detailed_costs,
            left_dists,
            right_dists,
        ) = py_planning_trees.plan_path(conesX, conesY, conesType)
        toc = time.time()
        print("{:.2f} ms".format((toc - tic) * 1000))
        waypoints = np.array(waypoints)
        path = np.array(path)

        if verbose:
            plt.clf()
            plt.scatter(conesX, conesY)
            plt.scatter(waypoints[:, 0], waypoints[:, 1])
            plt.plot(path[:, 0], path[:, 1])
            plt.savefig(f"./images/{self.counter}_main.png")
            for idx, test_path in enumerate(all_paths):
                test_path = np.array(test_path)
                plt.clf()
                plt.title(
                    "Total: {:.2f}, Path: {:.2f}, Color: {:.2f}, Angle: {:.2f}".format(
                        costs[idx], *detailed_costs[idx]
                    )
                )
                plt.scatter(conesX, conesY)
                plt.scatter(waypoints[:, 0], waypoints[:, 1])
                plt.plot(test_path[:, 0], test_path[:, 1])
                plt.savefig(f"./images/{self.counter}_test_{idx}.png")

        plt.clf()
        self.counter += 1
        return path


crns = np.array([False, True, True, False, True, True, True, False, True, True, False], dtype=bool)
delTh = np.array(
    [0, np.pi / 2, np.pi / 2, 0, np.pi / 2, np.pi / 2, np.pi / 2, 0, np.pi / 4, np.pi / 4, 0],
    dtype=float,
)
lpar = np.array([20, 10, -10, 20, 10, -10, 10, 200, -10, 10, 200], dtype=float)
track = Track(TrackConfig(), crns)
track.solve(lpar, delTh, case=0)
aveDist = 2
track.plot(cones=True, aveDist=aveDist)

simulator = Simulator(SimConfig(), track)
out = simulator.run(PlannerWrapper)

print(out, simulator.losses)
print(final_cones)

planner_wrapper = PlannerWrapper()
planner_wrapper.getPath(final_cones, True)
