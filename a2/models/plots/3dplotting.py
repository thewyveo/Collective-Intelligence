import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
'''

energy_data = pd.read_csv("merged_output.csv")

def prey_predator_number_plot(data):
    predator_x = data['predator'].values
    prey_y = data['prey'].values
    frames = data['frame'].values

    # Create segments
    points = np.array([predator_x, prey_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a LineCollection, colored by frame
    norm = plt.Normalize(frames.min(), frames.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(frames[:-1])  # Color based on frames
    lc.set_linewidth(2)

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.set_xlim(predator_x.min(), predator_x.max())
    ax.set_ylim(prey_y.min(), prey_y.max())

    plt.xlabel('Predator Population')
    plt.ylabel('Prey Population')
    plt.title('Prey-Predator Population Dynamics Colored by Time')
    plt.colorbar(lc, label='Frame (Time)')
    plt.grid()
    plt.tight_layout()
    #plt.show()
    plt.savefig('baseline_population_plot.png', dpi=300)


def prey_predator_frame_3d_plot(data):
    predator = data['predator'].values
    prey = data['prey'].values
    frames = data['frame'].values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(frames, prey, predator, linewidth=2)

    ax.set_xlabel('Hours (Frames)')
    ax.set_ylabel('Prey Population')
    ax.set_zlabel('Predator Population')
    ax.set_title('Prey–Predator Dynamics with Time')

    #plt.show()
    plt.savefig('baseline_3d_plot.png', dpi=300)

if __name__ == '__main__':
    prey_predator_number_plot(energy_data)
    prey_predator_frame_3d_plot(energy_data)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
data = pd.read_csv("merged_output.csv")
for run_id in data["run"].unique():
    run_df = data[data["run"] == run_id]
    plt.plot(run_df["frame"], run_df["prey"], color="green", alpha=0.3)
    plt.plot(run_df["frame"], run_df["predator"], color="red", alpha=0.3)

plt.title("Semi-Realistic Lotka-Volterra: 50 Runs")
plt.xlabel("Frame")
plt.ylabel("Population")
plt.grid(True)
plt.tight_layout()
plt.savefig("semi_realistic_50_runs.png")
plt.close()
print("Plot saved to semi_realistic_50_runs.png")
'''

def plot_species_3d_trajectory(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data["species_a"]
    y = data["species_b"]
    z = data["species_c"]
    t = data["frame"]

    # Optional: use time as color
    ax.plot(x, y, z, label="Population Trajectory", color="black", linewidth=2)
    sc = ax.scatter(x, y, z, c=t, cmap='viridis', s=2)

    ax.set_xlabel("Species A")
    ax.set_ylabel("Species B")
    ax.set_zlabel("Species C")
    ax.set_title("3D Species Dynamics (A-B-C)")
    fig.colorbar(sc, label="Time (Frame)")

    plt.tight_layout()
    plt.savefig("3species_abc_trajectory.png", dpi=300)
    plt.close()
    print("Saved: 3species_abc_trajectory.png")

data_w4 = pd.read_csv("rock-paper-scissors1.csv")
plot_species_3d_trajectory(data_w4)
