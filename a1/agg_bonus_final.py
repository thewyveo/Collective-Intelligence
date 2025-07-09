from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import polars as pl
import math
import matplotlib.pyplot as plt
import numpy as np
import os

WANDERING = 0
JOINING = 1
STILL = 2
LEAVING = 3
clusters = []
cluster_data = []  # Global list to store cluster data during simulation
frames = 0

@dataclass
class AggregationConfig(Config):
    p_join_base: float = 0.01
    p_leave_base: float = 0.05
    alpha: float = 0.4
    beta: float = 0.25

    window_size: tuple[int, int] = (750, 750)
    delta_time: float = 1.0
    movement_speed: float = 30
    radius: int = 30
    duration: int = 10000

    t_join: int = 3
    t_leave: int = 5
    check_interval: int = 1 * delta_time

class AggregationAgent(Agent):
    def on_spawn(self):
        self.state = WANDERING
        self.timer = 0
        self.check_timer = 0
        self.change_image(WANDERING)
        self.cluster_id = None

    def update(self):
        global frames, cluster_tracker
        
        # Track frame counter with first agent
        if self.id == 0:
            frames += 1
            if frames % 1000 == 0:
                print(f"Frame: {frames}/{self.config.duration}")
        
        self.timer += self.config.delta_time
        self.check_timer += self.config.delta_time
        neighbors = list(self.in_proximity_accuracy())
        n_still_neighbors = sum(1 for (a, _) in neighbors if a.state == STILL)
        dt = self.config.delta_time
        random_join_chance = 0.001

        if self.state == WANDERING:
            self.change_image(WANDERING)
            self.cluster_id = None  # Clear cluster ID when wandering
            
            if n_still_neighbors:
                lambda_stop = self.config.p_join_base + self.config.alpha * n_still_neighbors
                p_join = 1.0 - math.exp(-lambda_stop * dt)
                if random.random() < p_join:
                    self.state = JOINING
                    self.timer = 0
                    self.change_image(JOINING)

            elif random.random() < random_join_chance:
                self.state = JOINING
                self.timer = 0
                self.change_image(JOINING)

        elif self.state == JOINING:
            if self.timer >= self.config.t_join:
                self.state = STILL
                self.timer = 0
                self.check_timer = 0
                self.change_image(STILL)

                # Cluster management
                still_neighbors = [(a, d) for (a, d) in neighbors if a.state == STILL]
                if still_neighbors:
                    # Get all unique cluster IDs from neighbors
                    neighbor_clusters = set(a.cluster_id for (a, _) in still_neighbors if a.cluster_id is not None)
                    if neighbor_clusters:
                        # Join the smallest cluster ID
                        self.cluster_id = min(neighbor_clusters)
                        # Merge clusters by updating all neighbors to the same ID
                        for a, _ in still_neighbors:
                            if a.cluster_id in neighbor_clusters:
                                a.cluster_id = self.cluster_id
                    else:
                        # Create new cluster if no neighbors have clusters
                        self.create_cluster()
                else:
                    # Create new cluster if no still neighbors
                    self.create_cluster()

        elif self.state == STILL:
            if self.check_timer >= self.config.check_interval:
                self.check_timer = 0
                lambda_leave = self.config.p_leave_base * math.exp(-self.config.beta * n_still_neighbors)
                p_leave = 1.0 - math.exp(-lambda_leave * dt)
                if random.random() < p_leave:
                    self.state = LEAVING
                    self.timer = 0
                    self.change_image(WANDERING)

        elif self.state == LEAVING:
            if self.timer >= self.config.t_leave:
                self.state = WANDERING
                self.timer = 0
                self.change_image(WANDERING)
                self.cluster_id = None  # Leaving the cluster

        # Update cluster tracker every 10 frames
        if frames % 10 == 0 and self.cluster_id is not None and self.state == STILL:
            if frames not in cluster_tracker:
                cluster_tracker[frames] = {}
            if self.cluster_id not in cluster_tracker[frames]:
                cluster_tracker[frames][self.cluster_id] = 0
            cluster_tracker[frames][self.cluster_id] += 1

    def change_position(self):
        if self.there_is_no_escape():
            self.pos += self.move
        else:
            if self.state in [STILL, JOINING]:
                self.move = Vector2(0, 0)
            else:
                angle = random.uniform(0, 2 * math.pi)
                direction = Vector2(math.cos(angle), math.sin(angle))
                self.move = direction * self.config.movement_speed
            self.pos += self.move * self.config.delta_time

    def create_cluster(self):
        """Create a new cluster if the agent has no cluster_id"""
        if self.cluster_id is None:
            self.cluster_id = len(clusters) + 1
            clusters.append(self.cluster_id)

cfg = AggregationConfig()

print('Running simulation...')

clusters = []
cluster_tracker = {}
cluster_tracker_per_run = []
all_clusters = []
cluster_run = True
runs = 30
all_data = []

for i in range(runs):
    print(f"\nRun {i + 1}/{runs}")
    frames = 0
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

        cluster_tracker_per_run.append(cluster_tracker.copy())  # Save the tracker for analysis


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

# Final cluster statistics
biggest_cluster_sizes = []
cluster_counts = []

for tracker in cluster_tracker_per_run:
    if not tracker:
        continue
    last_frame = max(tracker.keys())
    final_clusters = tracker[last_frame]
    
    if final_clusters:
        cluster_counts.append(len(final_clusters))
        biggest_cluster_sizes.append(max(final_clusters.values()))
    else:
        cluster_counts.append(0)
        biggest_cluster_sizes.append(0)

print("\nCluster Statistics:")
print(cluster_counts)
print(biggest_cluster_sizes)
print(f'Average number of clusters per run: {np.mean(cluster_counts):.2f}')
print(f'Average size of biggest cluster per run: {np.mean(biggest_cluster_sizes):.2f}')