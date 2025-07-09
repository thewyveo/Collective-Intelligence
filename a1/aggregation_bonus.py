from dataclasses import dataclass
from vi import Agent, Config, Simulation
from pygame.math import Vector2
import random
import polars as pl
import seaborn as sns
import pygame as pg
import math

# States
WANDERING = 0
JOINING = 1
STILL = 2
LEAVING = 3
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
    duration: int = 5000

    t_join: int = 3
    t_leave: int = 5
    check_interval: int = 1

class AggregationAgent(Agent):

    def on_spawn(self):
        self.state = WANDERING
        self.timer = 0
        self.check_timer = 0
        self.change_image(WANDERING)

    def update(self):
        if self.id == 0:
            global frames
            frames += 1
            print(f"{frames-1}/5000")
        self.timer += self.config.delta_time
        self.check_timer += self.config.delta_time

        neighbors = list(self.in_proximity_accuracy())
        n_still_neighbors = sum(1 for (a, _) in neighbors if a.state == STILL)
        dt = self.config.delta_time

        random_join_chance = 0.001

        if self.state == WANDERING:
            self.change_image(WANDERING)

            if n_still_neighbors:
                # Positive feedback-based joining
                lambda_stop = self.config.p_join_base + self.config.alpha * n_still_neighbors
                p_join = 1.0 - math.exp(-lambda_stop * dt)
                if random.random() < p_join:
                    self.state = JOINING
                    self.timer = 0
                    self.change_image(JOINING)
            elif random.random() < random_join_chance:
                # Seed random clusters
                self.state = JOINING
                self.timer = 0
                self.change_image(JOINING)

        elif self.state == JOINING:
            if self.timer >= self.config.t_join:
                self.state = STILL
                self.timer = 0
                self.check_timer = 0
                self.change_image(STILL)

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

pg.font.init()
font = pg.font.SysFont("Arial", 24)

# Run the simulation
print("Running Aggregation Simulation WITHOUT Sites (True Emergent Clustering)")
cfg = AggregationConfig()
sim = (
    Simulation(cfg)
    .batch_spawn_agents(100, AggregationAgent, images=[
        "images/triangle.png",   # 0: Wandering
        "images/triangle2.png",  # 1: Joining
        "images/green.png",      # 2: Still
        "images/triangle.png"   # 3: Leaving
    ])
    .run()
)

# Process simulation data
print("Processing simulation data...")
data = (
    sim.snapshots
    .group_by(["frame", "image_index"])
    .agg(pl.count("id").alias("agents"))
    .sort(["frame", "image_index"])
)

plot = sns.relplot(
    data=data,
    x="frame",
    y="agents",
    hue="image_index",
    kind="line",
    palette=["blue", "orange", "green", "red"]
)
plot.savefig("experiment_emergent_true.png", dpi=300, bbox_inches="tight")
print("Simulation completed! Results saved to experiment_emergent_true.png")
