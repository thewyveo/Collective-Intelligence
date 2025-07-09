from dataclasses import dataclass, field
import platform, vi, jinja2, markupsafe, polars
from vi import Agent, Config, Simulation
from pygame.math import Vector2
import random
import polars as pl
import seaborn as sns
import pygame as pg
import math

# States as defined in the assignment
WANDERING = 0
JOINING = 1
STILL = 2
LEAVING = 3

@dataclass
class AggregationConfig(Config):
    # Probability parameters - these need to be tuned based on neighbor density
    p_join_base: float = 0.05      # Base probability to join when entering a site
    p_leave_base: float = 0.05     # Base probability to leave when in Still state

    # reinforcement / inhibition slopes from the paper
    alpha: float = 0.08         # α – positive feedback per neighbour (s⁻¹)
    beta:  float = 0.35         # β – negative feedback per neighbour (–)
    
    # Simulation parameters
    window_size: tuple[int, int] = (750, 750)
    delta_time: float = 10.0
    movement_speed: float = 10   # Speed when moving
    radius: int = 40               # Agent sensing radius
    duration: int = 1000           # Simulation duration

    # Timer parameters (in simulation steps)
    t_join: int = 5 * delta_time # Time to spend in Joining state before becoming Still
    t_leave: int = 5 * delta_time  # Time to spend in Leaving state before Wandering

    # Check interval for leave probability (every D time steps as per assignment)
    check_interval: int = 2 * delta_time

    #innitialise sites parameters
    sites: list[tuple[float, float, float]] = field(init=False)
    
    def __post_init__(self):
        w, h = self.window_size
        # For Stage 2 Experiment 1: Different sized sites (asymmetric aggregation)
        # Left site (smaller), Right site (larger)
        self.sites = [
            (w/3, h/2, 25),      # Left site: smaller radius
            ((w/3)*2, h/2, 75)   # Right site: larger radius
        ]

class AggregationAgent(Agent):

    """Positive‑/negative‑feedback aggregation after Garnier et al.

    * Variable names and overall PFSM skeleton kept from the user’s code.
    * The **probability laws** now follow
        λ_stop  = λ₀ + α n
        λ_leave = μ₀ e^(−β n)
      and are converted to per‑step probabilities with
        P = 1 − exp(−λ Δt).
    """


    def on_spawn(self):
        # All agents start in WANDERING state as per assignment
        self.state = WANDERING
        self.timer = 0
        self.check_timer = 0
        self.change_image(0)  # Wandering image
        
    def update(self):
        """Update agent state based on PFSM logic from assignment"""
        self.timer += 1 * self.config.delta_time
        self.check_timer += 1 * self.config.delta_time
        
        # Count neighbors within sensing radius
        neighbors = list(self.in_proximity_accuracy())
        n_neighbors = len(neighbors)
        dt = self.config.delta_time
        
        # State transitions based on assignment description
        if self.state == WANDERING:
            # Check if agent entered a site
            self.change_image(0)  # Wandering image
            on,site_pos = self.on_site()
            if on:
                # Calculate P_join based on neighbor density
                # More neighbors = higher probability to join
                lambda_stop = self.config.p_join_base + (self.config.alpha * n_neighbors)
                p_join = 1.0 -math.exp(-lambda_stop*dt)
                if random.random() < p_join:
                    self.state = JOINING
                    self.move_to_site(site_pos) # Move to site position
                    self.timer = 0
                    self.change_image(1)  # Joining image
                    
        elif self.state == JOINING:
            # Transition to STILL after T_join time steps
            if self.timer >= self.config.t_join:
                self.state = STILL
                self.timer = 0
                self.check_timer = 0
                self.change_image(2)  # Still image (green as per original)
                
        elif self.state == STILL:
            # Check leave probability every D time steps
            if self.check_timer >= self.config.check_interval:
                self.check_timer = 0
                
                # Calculate P_leave based on neighbor density
                # More neighbors = lower probability to leave
                lambda_leave = self.config.p_leave_base * math.exp(-self.config.beta * n_neighbors)
                p_leave = 1.0 - math.exp(-lambda_leave * dt)
                
                if random.random() < p_leave:
                    self.state = LEAVING
                    self.timer = 0
                    self.change_image(3)  # Leaving image
                    
        elif self.state == LEAVING:
            # Transition to WANDERING after T_leave time steps
            if self.timer >= self.config.t_leave:
                self.state = WANDERING
                self.timer = 0
                self.change_image(0)

    def change_position(self):
        if self.there_is_no_escape():
            self.pos += self.move
        else:
        
            if self.state == STILL:
                self.move = Vector2(0, 0)

            elif self.state == JOINING:
                on, site_pos = self.on_site()

                #if the agent is not on the site but detects some site within it's radius
                if not on and site_pos is not None:
                    self.move = self.move_to_site(site_pos)  # keep steering in
                else:
                    self.move = Vector2(0, 0)

            elif self.state in [WANDERING, LEAVING]:
                # Slight randomness for all moving states
                random_angle = random.uniform(0, 2 * 3.14159)
                random_vector = Vector2(1, 0).rotate_rad(random_angle)
                self.move += random_vector * 0.2  # Scale down the randomness

            self.pos += self.move * self.config.delta_time

    def on_site(self)-> tuple[bool,Vector2]:
        """Check if the agent is on any aggregation site"""
        """Return (True, site_pos) if inside any patch, else (False, None)."""
        for x, y, radius in self.config.sites:
            site_pos = Vector2(x, y)
            if self.pos.distance_to(site_pos) < radius:
                return True, site_pos
        return False, None

    def move_to_site(self, site_pos):
        """Move towards the aggregation site"""
        direction = (site_pos - self.pos).normalize()
        self.move = direction * self.config.movement_speed
        return self.move

# Stage 2 - Experiment 1: Different sized aggregation sites
print("Running Stage 2 - Experiment 1: Different sized sites")
cfg1 = AggregationConfig()
w, h = cfg1.window_size

sim1 = (
    Simulation(cfg1)
    .batch_spawn_agents(100, AggregationAgent, images=[
        "images/triangle.png",  # 0: Wandering
        "images/triangle2.png",  # 1: Joining  
        "images/green.png",     # 2: Still
        "images/triangle3.png"   # 3: Leaving
    ])
    .spawn_site("images/circle.png", x=w/3, y=h/2)
    .spawn_site("images/circle2.png", x=(w/3)*2, y=h/2)
    .run()
)
print(f"c x = {w/3}, y = {h/2}")
print(f"c2 x = {(w/3)*2}, y = {h/2}")
# Process and visualize results
print("Processing simulation data...")
data1 = (
    sim1.snapshots
    .group_by(["frame", "image_index"])
    .agg(pl.count("id").alias("agents"))
    .sort(["frame", "image_index"])
)

# Create visualization
plot1 = sns.relplot(
    data=data1, 
    x="frame", 
    y="agents", 
    hue="image_index", 
    kind="line",
    palette=["blue", "orange", "green", "red"]
)
plot1.savefig("experiment1_different_sizes.png", dpi=300, bbox_inches='tight')
print("Experiment 1 completed! Results saved to experiment1_different_sizes.png")





# Stage 2 - Experiment 2: Same sized aggregation sites
print("\nRunning Stage 2 - Experiment 2: Same sized sites")


@dataclass
class AggregationConfig2(Config):
    # Probability parameters - these need to be tuned based on neighbor density
    p_join_base: float = 0.05  # Base probability to join when entering a site
    p_leave_base: float = 0.05  # Base probability to leave when in Still state

    # reinforcement / inhibition slopes from the paper
    alpha: float = 0.08  # α – positive feedback per neighbour (s⁻¹)
    beta: float = 0.35  # β – negative feedback per neighbour (–)

    # Simulation parameters
    window_size: tuple[int, int] = (750, 750)
    delta_time: float = 10.0
    movement_speed: float = 15  # Speed when moving
    radius: int = 40  # Agent sensing radius
    duration: int = 1000  # Simulation duration

    # Timer parameters (in simulation steps)
    t_join: int = 5 * delta_time  # Time to spend in Joining state before becoming Still
    t_leave: int = 5 * delta_time  # Time to spend in Leaving state before Wandering

    # Check interval for leave probability (every D time steps as per assignment)
    check_interval: int = 2 * delta_time

    # innitialise sites parameters
    sites: list[tuple[float, float, float]] = field(init=False)

    def __post_init__(self):
        w, h = self.window_size
        # For Stage 2 Experiment 1: Different sized sites (asymmetric aggregation)
        # Left site (smaller), Right site (larger)
        self.sites = [
            (w // 4, h // 2, 75),  # Left site: smaller radius
            ((3*w // 4), h // 2, 75)  # Right site: larger radius
        ]


cfg2 = AggregationConfig2(Config)

sim2 = (
    Simulation(cfg2)
    .batch_spawn_agents(100, AggregationAgent, images=[
        "images/triangle.png",  # 0: Wandering
        "images/triangle2.png",  # 1: Joining  
        "images/green.png",     # 2: Still
        "images/triangle3.png"   # 3: Leaving
    ])
    .spawn_site("images/circle2.png", x=w//4, y=h//2)      # Left site
    .spawn_site("images/circle2.png", x=3*w//4, y=h//2)    # Right site
    .run()
)

# ---------------- after sim2.run() -----------------
print("Processing Experiment-2 data…")

# 2️⃣  build Polars table
data2 = (
    sim2.snapshots
        .group_by(["frame", "image_index"])
        .agg(pl.count("id").alias("agents"))
        .sort(["frame", "image_index"])
)

# 3️⃣  draw & save the figure
make_state_plot(
    data2,
    png_name="experiment2_same_sizes.png",
    title="Experiment 2 – Same-sized sites",
)

print("Experiment 2 completed! Results saved to experiment2_same_sizes.png")



print("\nBoth experiments completed successfully!")
print("Expected behavior:")
print("- Experiment 1: Agents should eventually aggregate more in the larger site")
print("- Experiment 2: Agents should eventually converge to a single site despite equal sizes")