from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import polars as pl
import math
import matplotlib.pyplot as plt
from collections import defaultdict

WANDERING = 0
JOINING = 1
STILL = 2
LEAVING = 3
frames = 0
clusters = []

@dataclass
class AggregationConfig(Config):
    p_join_base: float = 0.01  # 0.01
    p_leave_base: float = 0.05  # 0.05
    alpha: float = 0.4  # 0.4
    beta: float = 0.25  # 0.25

    window_size: tuple[int, int] = (750, 750)
    delta_time: float = 1.0
    movement_speed: float = 30  # 30
    radius: int = 30
    duration: int = 10000

    t_join: int = 3 # 3
    t_leave: int = 5    # 5
    check_interval: int = 1 * delta_time  # 1 * delta_time

cluster_counter = 0  # global counter for cluster IDs

class AggregationAgent(Agent):
    def on_spawn(self):
        self.state = WANDERING
        self.timer = 0
        self.check_timer = 0
        self.cluster_id = None  # for cluster tracking
        self.change_image(WANDERING)

    def assign_cluster_id(self, neighbors):
        global cluster_counter
        neighbor_cluster_ids = [
            a.cluster_id for (a, _) in neighbors
            if a.state == STILL and a.cluster_id is not None
        ]
        if neighbor_cluster_ids:
            # Join an existing cluster (use min to merge overlapping ones)
            self.cluster_id = min(neighbor_cluster_ids)
        else:
            # Create a new cluster
            cluster_counter += 1
            self.cluster_id = cluster_counter

    def update(self):
        if self.id == 0:
            global frames
            frames += 1
            if frames % 100 == 0:
                print(f"{frames}/{self.config.duration}")

        self.timer += self.config.delta_time
        self.check_timer += self.config.delta_time

        neighbors = list(self.in_proximity_accuracy())
        n_still_neighbors = sum(1 for (a, _) in neighbors if a.state == STILL)
        dt = self.config.delta_time
        random_join_chance = 0.001

        if self.state == WANDERING:
            self.change_image(WANDERING)
            self.cluster_id = None  # left cluster

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
                if n_still_neighbors > 7:
                    if frames > 500:
                        global clusters, test
                        self.assign_cluster_id(neighbors)  # assign when entering STILL
                        clusters.append((frames, self.cluster_id, test))

        elif self.state == STILL:
            if self.check_timer >= self.config.check_interval:
                self.check_timer = 0
                lambda_leave = self.config.p_leave_base * math.exp(-self.config.beta * n_still_neighbors)
                p_leave = 1.0 - math.exp(-lambda_leave * dt)
                if random.random() < p_leave:
                    self.state = LEAVING
                    self.timer = 0
                    self.change_image(WANDERING)
                    self.cluster_id = None  # leaving, untag from cluster

        elif self.state == LEAVING:
            if self.timer >= self.config.t_leave:
                self.state = WANDERING
                self.timer = 0
                self.change_image(WANDERING)
                self.cluster_id = None  # fully left, untag

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


cfg = AggregationConfig()

runs = 30
test = 1
all_data = []
for run in range(runs):
    print(f"\nRun {run + 1}/{runs}"*3)
    print()
    frames = 0
    test += 1
    cluster_counter = 0
    sim = (
        HeadlessSimulation(cfg)
        .batch_spawn_agents(100, AggregationAgent, images=[
            "images/triangle.png",   # 0: wandering
            "images/triangle2.png",  # 1: joining
            "images/green.png",      # 2: still
            "images/triangle.png"   # 3: leaving
        ])
        .run()
    )

    raw_data = sim.snapshots.to_pandas()
    
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
    grouped['run'] = run
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

final = []
clusters.sort(key=lambda x: x[2])  # Sort clusters by 'test/run'
all_clusters = defaultdict(list)
for (frames, cluster_id, run_id) in clusters:
    all_clusters[int(run_id-1)].append((frames, cluster_id))

for key, value in all_clusters.items():
    value.sort(key=lambda x: x[0])  # sort by frame
    value.reverse()
    for (frame, cluster_id) in value:
        if frame > 9000:
            final.append((frame, cluster_id, str(key)))


results = []
for (frame, cluster_id, run_id) in final:
    if cluster_id not in [r[1] for r in results if r[1] == run_id]:
        results.append((frame, cluster_id, run_id))
print(results)

'''
for j in range(runs):
    # Get all entries for this run
    run_entries = [row for row in final if row[2] == str(j)]

    if not run_entries:
        results.append((0, j))  # no clusters for this run
        continue

    prev_cluster_id = run_entries[0][1]
    seen_clusters = set()
    ctr = 0

    for i in range(1, len(run_entries)):
        current_cluster_id = run_entries[i][1]
        if current_cluster_id != prev_cluster_id and prev_cluster_id not in seen_clusters:
            ctr += 1
            seen_clusters.add(prev_cluster_id)
        prev_cluster_id = current_cluster_id

    results.append((ctr, j))


print(results)
'''
ax.set_xlabel('Frame')
ax.set_ylabel('Number of Agents')
ax.set_title(f'Agent States Over Time (Mean ± Std Dev) for {runs} Runs')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experimentB_no_sites.png", dpi=300, bbox_inches='tight')
print("Combined plot saved to experimentB_no_sites.png")

# Save statistics
with open("experimentB_no_sites.txt", "w") as f:
    f.write("frame,mean_wandering,std_wandering,mean_joining,std_joining,mean_still,std_still,\n")
    for row in stats.rows():
        f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}\n")

print("Statistics saved to experimentB_no_sites.txt")