from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import polars as pl
import math
import matplotlib.pyplot as plt
import numpy as np
import os


RANDOM_WALK = 0
APPROACH = 1
WAIT = 2
LEAVING = 3
frames = 0
clusters = []
cluster_data = []

@dataclass
class AggregationConfig(Config):
    G_leave: float = 0.5
    leave_exponent: float = 2.5
    rw_time: float = 20.0

    window_size: tuple[int, int] = (750, 750)
    delta_time: float = 1.0
    movement_speed: float = 10
    radius: int = 30
    duration: int = 10000

    long_range: float = 45
    contact_range: float = 45

    t_leave: int = 10
    t_join: int = 3
    random_join_chance: float = 0.001

class AggregationAgent(Agent):
    def _set_state(self, new_state: int):
        if new_state != getattr(self, "state", None):
            self.state = new_state
            self.change_image(new_state)
            self.timer = 0

    def on_spawn(self):
        self._set_state(RANDOM_WALK)
        self.timer = 0
        self.cluster_id = None

    def update(self):
        global frames
        if self.id == 0:
            frames += 1
            if frames % 100 == 1:
                print(f"frame {frames-1}/{self.config.duration}")

        neighbors = list(self.in_proximity_accuracy())
        n_still_neighbors = sum(1 for (a, _) in neighbors if a.state == WAIT)
        dt = self.config.delta_time
        self.timer += dt

        if self.state == RANDOM_WALK:
            self.change_image(RANDOM_WALK)
            self.cluster_id = None

            if n_still_neighbors and not self.state == LEAVING:
                self._set_state(APPROACH)
            elif random.random() < self.config.random_join_chance:
                self._set_state(WAIT)

        elif self.state == APPROACH:
            self.change_image(APPROACH)
            if n_still_neighbors:
                centroid = sum((a.pos for (a, _) in neighbors if a.state == WAIT), Vector2()) / n_still_neighbors
                if centroid.distance_to(self.pos) <= self.config.contact_range:
                    self._set_state(WAIT)

                    # Cluster management
                    wait_neighbors = [(a, d) for (a, d) in neighbors if a.state == WAIT]
                    if wait_neighbors:
                        # Get all unique cluster IDs from neighbors
                        neighbor_clusters = set(a.cluster_id for (a, _) in wait_neighbors if a.cluster_id is not None)
                        if neighbor_clusters:
                            # Join the smallest cluster ID
                            self.cluster_id = min(neighbor_clusters)
                            # Merge clusters by updating all neighbors to the same ID
                            for a, _ in wait_neighbors:
                                if a.cluster_id in neighbor_clusters:
                                    a.cluster_id = self.cluster_id
                        else:
                            # Create new cluster if no neighbors have clusters
                            self.create_cluster()
                    else:
                        # Create new cluster if no wait neighbors
                        self.create_cluster()

            else:
                self._set_state(LEAVING)

        elif self.state == WAIT:
            cluster_size = n_still_neighbors + 1
            lambd = self.config.G_leave / (cluster_size ** self.config.leave_exponent)
            Pstep = 1 - math.exp(-lambd * dt)
            if random.random() < Pstep:
                if cluster_size == 1:
                    escape_dir = Vector2(1, 0).rotate_rad(random.uniform(0, 2 * math.pi))
                else:
                    centroid = sum((a.pos for (a, _) in neighbors if a.state == WAIT), Vector2()) / n_still_neighbors
                    escape_dir = (self.pos - centroid).normalize()
                self.move = escape_dir * self.config.movement_speed
                self._set_state(LEAVING)

        elif self.state == LEAVING:
            if self.timer >= self.config.t_leave:
                self._set_state(RANDOM_WALK)
                self.cluster_id = None  # Leaving the cluster

        # Update cluster tracker every 10 frames
        if frames % 10 == 0 and self.cluster_id is not None and self.state == WAIT:
            if frames not in cluster_tracker:
                cluster_tracker[frames] = {}
            if self.cluster_id not in cluster_tracker[frames]:
                cluster_tracker[frames][self.cluster_id] = 0
            cluster_tracker[frames][self.cluster_id] += 1

    def change_position(self):
        if self.there_is_no_escape():
            self.move.scale_to_length(self.config.movement_speed)
            self.pos += self.move * self.config.delta_time
        else:
            if self.state == WAIT:
                self.move = Vector2(0, 0)
            elif self.state == APPROACH:
                neighbors = list(self.in_proximity_accuracy())
                n_still_neighbors = sum(1 for (a, _) in neighbors if a.state == WAIT)
                if n_still_neighbors:
                    centroid = sum((a.pos for (a, _) in neighbors if a.state == WAIT), Vector2()) / n_still_neighbors
                    direction = (centroid - self.pos).normalize()
                    self.move = direction * self.config.movement_speed
                else:
                    self.move += Vector2(1, 0).rotate_rad(random.uniform(0, 2 * math.pi)) * 0.3
            else:
                self.move += Vector2(1, 0).rotate_rad(random.uniform(0, 2 * math.pi)) * 0.3

            if self.move.length_squared() > 0:
                self.move.scale_to_length(self.config.movement_speed)
            self.pos += self.move * self.config.delta_time

    def create_cluster(self):
        """Create a new cluster if the agent has no cluster_id"""
        if self.cluster_id is None:
            self.cluster_id = len(clusters) + 1
            clusters.append(self.cluster_id)

cfg = AggregationConfig()

print('Running simulation...')

all_clusters = []
cluster_run = True
runs = 30
all_data = []

for i in range(runs):
    print(f"\nRun {i + 1}/{runs}")
    frames = 0
    clusters = []
    cluster_tracker = {}

    sim = (
        HeadlessSimulation(cfg)
        .batch_spawn_agents(100, AggregationAgent, images=[
            "images/triangle.png",
            "images/triangle2.png",
            "images/green.png",
            "images/triangle.png"
        ])
        .run()
    )

    last_frame = max(cluster_tracker.keys())
    final_clusters = cluster_tracker[last_frame]
    num_clusters = {}

    for cluster_id, size in final_clusters.items():
        num_clusters[cluster_id] = size

    all_clusters.append(num_clusters)

    raw_data = sim.snapshots.to_pandas()

    if cluster_run:
        print("Cluster data collected during the run.")
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

        def plot_state():
            grouped = (
                raw_data
                .groupby(['frame'])
                .agg(
                    wandering=('image_index', lambda x: (x == 0).sum()),
                    joining=('image_index', lambda x: (x == 1).sum()),
                    still=('image_index', lambda x: (x == 2).sum()),
                    leaving=('image_index', lambda x: (x == 3).sum())
                )
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(12, 8))

            ax.plot(grouped['frame'], grouped['wandering'], label="Wandering", color="blue")
            ax.plot(grouped['frame'], grouped['joining'], label="Joining", color="orange")
            ax.plot(grouped['frame'], grouped['still'], label="Still", color="green")
            ax.plot(grouped['frame'], grouped['leaving'], label="Leaving", color="purple")

            ax.set_xlabel('Frame')
            ax.set_ylabel('Number of Agents')
            ax.set_title(f'Agent States Over Time (Single Run)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "experimentB_single_run.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to {os.path.join(plots_dir, 'experimentB_single_run.png')}")

        def plot_cluster_lifecycles():
            """Plot cluster lifecycles only"""
            if not cluster_tracker:
                print("No cluster data collected!")
                return
            
            all_cluster_ids = set()
            for clusters_dict in cluster_tracker.values():
                all_cluster_ids.update(clusters_dict.keys())
            unique_clusters = sorted(all_cluster_ids)
            
            n_colors = max(len(unique_clusters), 10)
            colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for i, cluster_id in enumerate(unique_clusters):
                frames = []
                present = []
                
                for frame in sorted(cluster_tracker.keys()):
                    if cluster_id in cluster_tracker[frame]:
                        frames.append(frame)
                        present.append(cluster_tracker[frame][cluster_id])
                
                if frames:
                    avg_size = sum(present) / len(present)
                    line_width = max(1, min(8, avg_size / 3))
                    
                    jitter = np.random.uniform(-0.3, 0.3)  # small vertical offset
                    ax.plot(frames, [cluster_id + jitter] * len(frames), 
                            color=colors[i % len(colors)], 
                            linewidth=line_width, 
                            alpha=0.8)
            
            ax.set_xlabel('Frame')
            ax.set_ylabel('Cluster ID')
            ax.set_title('Cluster Lifecycles\n(Line thickness represents average size)')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "cluster_lifecycles.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Cluster lifecycles plot saved to {os.path.join(plots_dir, 'cluster_lifecycles.png')}")

        def plot_cluster_sizes():
            """Plot cluster sizes over time"""
            if not cluster_tracker:
                print("No cluster data collected!")
                return
            
            all_cluster_ids = set()
            for clusters_dict in cluster_tracker.values():
                all_cluster_ids.update(clusters_dict.keys())
            unique_clusters = sorted(all_cluster_ids)
            
            n_colors = max(len(unique_clusters), 10)
            colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            for i, cluster_id in enumerate(unique_clusters):
                frames = []
                sizes = []
                
                for frame in sorted(cluster_tracker.keys()):
                    if cluster_id in cluster_tracker[frame]:
                        frames.append(frame)
                        sizes.append(cluster_tracker[frame][cluster_id])
                
                if frames:
                    max_size = max(sizes)
                    ax.plot(frames, sizes, 
                            color=colors[i % len(colors)], 
                            label=f'Cluster {cluster_id}' if max_size > 10 else "_nolegend_",
                            linewidth=1.5,
                            marker='.',
                            markersize=2)

            ax.set_xlabel('Frame')
            ax.set_ylabel('Cluster Size')
            ax.set_title('Cluster Sizes Over Time')
            ax.grid(True, alpha=0.3)
            
            # Add legend outside of plot
            ax.legend(bbox_to_anchor=(1.05, 1),
                    loc='upper left', 
                    borderaxespad=0.
                    )
            
            plt.savefig(os.path.join(plots_dir, "cluster_sizes.png"), 
                        dpi=300, 
                        bbox_inches='tight',
                        pad_inches=0.5)
            plt.close()
            print(f"Cluster sizes plot saved to {os.path.join(plots_dir, 'cluster_sizes.png')}")
            
            # Print statistics
            total_datapoints = sum(len(clusters) for clusters in cluster_tracker.values())
            avg_size = sum(size for clusters in cluster_tracker.values() 
                        for size in clusters.values()) / total_datapoints if total_datapoints else 0
            max_size = max(max(clusters.values()) for clusters in cluster_tracker.values())
            
            print(f"\nCluster Summary:")
            print(f"Total unique clusters: {len(unique_clusters)}")
            print(f"Total data points: {total_datapoints}")
            print(f"Average cluster size: {avg_size:.2f}")
            print(f"Maximum cluster size: {max_size}")

        def plot_active_clusters():
            """Plot number of active clusters over time"""
            if not cluster_tracker:
                print("No cluster data for active clusters plot!")
                return
            
            frames_list = sorted(cluster_tracker.keys())
            n_clusters_list = [len(cluster_tracker[frame]) for frame in frames_list]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(frames_list, n_clusters_list, color='red', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Frame', fontsize=12)
            ax.set_ylabel('Number of Active Clusters', fontsize=12)
            ax.set_title('Number of Active Clusters Over Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "active_clusters.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Active clusters plot saved to {os.path.join(plots_dir, 'active_clusters.png')}")

        plot_state()
        plot_cluster_lifecycles()
        plot_cluster_sizes()
        plot_active_clusters()

        print(f"\nAll plots have been saved to the '{plots_dir}' directory.")

        cluster_run = False

    grouped = (
        raw_data
        .groupby(['frame'])
        .agg(
            wandering=('image_index', lambda x: (x == 0).sum()),
            joining=('image_index', lambda x: (x == 1).sum()),
            still=('image_index', lambda x: (x == 2).sum()),
            leaving=('image_index', lambda x: (x == 3).sum())
        )
        .reset_index()
    )
    grouped['run'] = runs
    data_run = pl.from_pandas(grouped)
    all_data.append(data_run)

combined_data = pl.concat(all_data)

stats = (
    combined_data
    .group_by(["frame"])
    .agg([
        pl.col("wandering").mean().alias("mean_wandering"),
        pl.col("wandering").std().alias("std_wandering"),
        pl.col("joining").mean().alias("mean_joining"),
        pl.col("joining").std().alias("std_joining"),
        pl.col("still").mean().alias("mean_still"),
        pl.col("still").std().alias("std_still"),
        pl.col("leaving").mean().alias("mean_leaving"),
        pl.col("leaving").std().alias("std_leaving")
    ])
    .sort(["frame"])
)

stats_pd = stats.to_pandas()

fig, ax = plt.subplots(figsize=(12, 8))

states = [
    ("Wandering", "mean_wandering", "std_wandering", "blue"),
    ("Joining", "mean_joining", "std_joining", "orange"),
    ("Still", "mean_still", "std_still", "green"),
    ("Leaving", "mean_leaving", "std_leaving", "purple")
    ]

for label, mean_col, std_col, color in states:
    ax.plot(stats_pd['frame'], stats_pd[mean_col], 
            color=color, label=label, linewidth=2)
    ax.fill_between(stats_pd['frame'],
                    stats_pd[mean_col] - stats_pd[std_col],
                    stats_pd[mean_col] + stats_pd[std_col],
                    color=color, alpha=0.2)

ax.set_xlabel('Frame')
ax.set_ylabel('Number of Agents')
ax.set_title(f'Agent States Over Time (Mean ± Std Dev) for {runs} Runs')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(plots_dir, "experimentB.png"), 
                        dpi=300, 
                        bbox_inches='tight'
                        )

# Save statistics
with open("plots/experimentB.txt", "w") as f:
    f.write("frame,mean_wandering,std_wandering,mean_joining,std_joining,mean_still,std_still,\n")
    for row in stats.rows():
        f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}\n")

# Print final cluster statistics
print("\nCluster Statistics:")
print(f'All clusters: {all_clusters}')
print(f'average number of clusters per run: {np.mean([len(c) for c in all_clusters])}')
print(f'average agent number in biggest cluster for each run.: {np.mean([max(c.values()) for c in all_clusters])}')