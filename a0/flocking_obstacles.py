from dataclasses import dataclass
import platform, vi, jinja2, markupsafe, polars
from vi import Agent, Config, Simulation, Matrix
from pygame.math import Vector2
import sys
import polars as pl             
from typing import Generator, Self
import pygame as pg
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool

import random
iters = 0
TURN = 0.05

@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 0.5
    cohesion_weight: float = 0.5
    separation_weight: float = 0.5
    delta_time: float = 3
    mass: int = 20
    window_size: tuple[int, int] = (750, 750)
    record_snapshots: bool = True
    

    def weights(self) -> tuple[float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight)

class Window:
    width: int = 750
    height: int = 750
        
class FlockingAgent(Agent):

    def update(self) -> None:
        if any(self.in_proximity_accuracy()):
            self.change_image(1)
        else:
            self.change_image(0)
     
    
    def change_position(self):
        self.there_is_no_escape()


        neighbors = list(self.in_proximity_accuracy())  # get neighbors in proximity
        global iters
                # global frame counter & hard stop
        if self.id == 0:                   # only one agent does the counting
            iters += 1
            print('iteration #', iters)
                
        
            
        if neighbors:

            alignment = Vector2(0, 0)
            cohesion = Vector2(0, 0)
            separation = Vector2(0, 0)

            self_pos = self.pos     # get self postiion & velocity
            self_vel = self.move
            n = len(neighbors)
            

            for other, _ in neighbors:
                other_pos = other.pos
                other_vel = other.move
                d = other_pos.distance_to(self_pos) or 1.0
                d = max(d, 5)

                alignment += other_vel  # add velocities of each neighbor to alignment (to be divided by n later for average)
                cohesion += other_pos   # add positions of each neighbor to cohesion (to be divided by n later for average)
                separation += (self_pos - other_pos) # add the difference between self position and neighbor position to separation (to be divided by n later for average)

            alignment = ((alignment / n) - self_vel)    # Vn - Vboid where Vn is the avg. velocities of the neighbors (alignment/n)
            cohesion = ((cohesion / n) - self_pos)   # fc - Vboid where fc = Xn - Xboid where Xn is the avg. positions of neighbors (cohesion/n)
            separation = (separation/n) # the mean of the differences between boid's position and neighbors' positions

            #normalizing
            if alignment.length() > 0:  # avoid normalizing zero vectors (NaN values)
                    alignment = alignment.normalize()
            if cohesion.length() > 0:
                    cohesion = cohesion.normalize()
            if separation.length() > 0:
                    separation = separation.normalize()

            alignment_weight, cohesion_weight, separation_weight = self.config.weights()
            total_force = (
                alignment * alignment_weight
                + cohesion * cohesion_weight
                + separation * separation_weight
            ) / self.config.mass        # "ftotal = (alpha*alignment + beta*cohesion + gamma*separation) / mass" from Assignment_0.pdf


            self.move += total_force * self.config.delta_time


        # Obstacle Avoidance
        obstacle_hit = pg.sprite.spritecollideany(
            self,  # type: ignore[reportArgumentType]
            self._obstacles,
            pg.sprite.collide_mask,
        )

        collision = bool(obstacle_hit)

        # Reverse direction when colliding with an obstacle.
        if collision:
            self.move.rotate_ip(180)

        
        else:       # NO NEIGHBORS NEARBY
            # random walk if no neighbors
            random_angle = random.uniform(0, 2 * 3.14159)
            random_vector = Vector2(1, 0).rotate_rad(random_angle)
            self.move += random_vector * 0.2  # Scale down the randomness

        # update move
        max_speed = 3
        if self.move.length() > max_speed:
               self.move = self.move.normalize() * max_speed

            # update pos
        self.pos += self.move * self.config.delta_time




# Simulation launch

cfg = FlockingConfig(
        image_rotation=True,
        movement_speed=1,
        radius=50,
        window_size=(750, 750),
        duration = 500 # set window to 750×750
    )
w, h = cfg.window_size

sim = (
        Simulation(cfg
            )
        .batch_spawn_agents(100, FlockingAgent, images=["images/triangle.png", "images/green.png"]
            )

        .spawn_obstacle("images/triangle@200px.png", w // 2, h // 2)
        .spawn_obstacle("images/triangle@50px.png", w // 4, h // 4)
        .spawn_obstacle("images/triangle@50px.png", 3*w // 4, h // 4)
        .spawn_obstacle("images/triangle@50px.png", 3*w // 4, 3*h // 4)
        .spawn_obstacle("images/triangle@50px.png", w // 4, 3*h // 4)
        .run()
        .snapshots.write_parquet("agents.parquet")
        .groupby (["frame", "image_index"])
        .agg(pl.count("id").alias("agents"))
        .sort(["frame","image_index"])
        
        )



print(sim)

plot = sns.relplot(x=sim["frame"],y = sim["agents_obstacles_graph"], hue = sim["image_index"], kind ="line")
plot.savefig("agents.png", dpi = 300)



