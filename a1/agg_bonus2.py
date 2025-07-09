import matplotlib.pyplot as plt
from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import polars as pl
import pygame as pg
import math

RANDOM_WALK = 0
APPROACH = 1
WAIT = 2
LEAVING = 3
_frames = 0

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

    def update(self):
        global _frames
        if self.id == 0:
            _frames += 1
            if _frames % 100 == 1:
                print(f"frame {_frames-1}/{self.config.duration}")

        neighbors = list(self.in_proximity_accuracy())
        n_still_neighbors = sum(1 for (a, _) in neighbors if a.state == WAIT)
        dt = self.config.delta_time
        self.timer += dt

        if self.state == RANDOM_WALK:
            self.change_image(RANDOM_WALK)
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

# ---------- Run multiple simulations and collect stats ----------
runs = 1
cfg = AggregationConfig()
all_data = []

for run in range(runs):
    print(f"\n[Run {run + 1}/{runs}]")
    _frames = 0

    sim = (
        Simulation(cfg)
        .batch_spawn_agents(
            100,
            AggregationAgent,
            images=[
                "images/triangle.png",  # 0: RANDOM_WALK
                "images/triangle2.png", # 1: APPROACH
                "images/green.png",     # 2: WAIT
                "images/triangle.png"   # 3: LEAVING
            ],
        )
        .run()
    )

    raw_data = sim.snapshots.to_pandas()
    grouped = (
        raw_data
        .groupby(['frame'])
        .agg(
            random_walk=('image_index', lambda x: (x == 0).sum()),
            approach=('image_index', lambda x: (x == 1).sum()),
            wait=('image_index', lambda x: (x == 2).sum()),
            leaving=('image_index', lambda x: (x == 3).sum())
        )
        .reset_index()
    )
    grouped['run'] = run
    all_data.append(pl.from_pandas(grouped))

combined_data = pl.concat(all_data)

stats = (
    combined_data
    .group_by("frame")
    .agg([
        pl.col("random_walk").mean().alias("mean_rw"),
        pl.col("random_walk").std().alias("std_rw"),
        pl.col("approach").mean().alias("mean_approach"),
        pl.col("approach").std().alias("std_approach"),
        pl.col("wait").mean().alias("mean_wait"),
        pl.col("wait").std().alias("std_wait"),
        pl.col("leaving").mean().alias("mean_leaving"),
        pl.col("leaving").std().alias("std_leaving"),
    ])
    .sort("frame")
)

# ---------- Plotting ----------
df = stats.to_pandas()
fig, ax = plt.subplots(figsize=(12, 8))

state_map = [
    ("Random Walk", "mean_rw", "std_rw", "blue"),
    ("Approach", "mean_approach", "std_approach", "orange"),
    ("Wait", "mean_wait", "std_wait", "green"),
    ("Leaving", "mean_leaving", "std_leaving", "red"),
]

for label, mean_col, std_col, color in state_map:
    ax.plot(df["frame"], df[mean_col], label=label, color=color)
    ax.fill_between(df["frame"], df[mean_col] - df[std_col], df[mean_col] + df[std_col], color=color, alpha=0.2)

ax.set_title(f"Aggregation State Dynamics (Mean ± Std Dev) over {runs} Runs")
ax.set_xlabel("Frame")
ax.set_ylabel("Number of Agents")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("experimentB2_garnier.png", dpi=300)
print("Saved to experimentB2_garnier.png")

# Save statistics
with open("experimentB2_garnier.txt", "w") as f:
    f.write("frame,mean_wandering,std_wandering,mean_joining,std_joining,mean_still,std_still,\n")
    for row in stats.rows():
        f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}\n")

print("Statistics saved to experimentB2_garnier.txt")