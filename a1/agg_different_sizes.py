from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import polars as pl
import matplotlib.pyplot as plt
import math

WANDERING = 0
JOINING = 1
STILL = 2
LEAVING = 3
_frames = 0

@dataclass
class AggregationConfig(Config):
    p_join_base: float = 0.05      # base probability to join when entering a site
    p_leave_base: float = 0.05     # base probability to leave when in still state

    alpha: float = 0.05         # α – positive feedback per neighbour (s⁻¹)
    beta:  float = 0.35         # β – negative feedback per neighbour (–)
    
    window_size: tuple[int, int] = (750, 750)
    delta_time: float = 1.0
    movement_speed: float = 50
    radius: int = 30
    duration: int = 5000

    t_join: int = 5 * delta_time    # time to spend in joining state before becoming still
    t_leave: int = 5 * delta_time   # time to spend in leaving state before wandering

    check_interval: int = 2 * delta_time

    def __post_init__(self):
        """Initialize aggregation sites with different sizes."""
        w, h = self.window_size
        self.sites = [
            (w/3, h/2, 25),      # Site 0: smaller site
            ((w/3)*2, h/2, 75)   # Site 1: larger site
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
        self.state = WANDERING
        self.timer = 0
        self.check_timer = 0
        self.site_id = -1  # -1 means not associated with any site
        self.change_image(0)
        
    def update(self):
        """Update agent state based on PFSM logic from assignment"""
        self.timer += 1 * self.config.delta_time
        self.check_timer += 1 * self.config.delta_time
        
        neighbors = list(self.in_proximity_accuracy())
        n_neighbors = len(neighbors)
        dt = self.config.delta_time

        if self.id == 0:
            global _frames
            _frames += 1
            #print(_frames/self.config.duration)
        
        if self.state == WANDERING:
            self.change_image(0)
            on,site_pos = self.on_site()
            if on:
                lambda_stop = self.config.p_join_base + (self.config.alpha * n_neighbors)   # more neighbors = higher join probability
                p_join = 1.0 -math.exp(-lambda_stop*dt)
                if random.random() < p_join:
                    self.state = JOINING
                    self.move_to_site(site_pos) 
                    self.timer = 0
                    self.change_image(1)  
                    
        elif self.state == JOINING:
            if self.timer >= self.config.t_join:    # after T time steps go to still state
                self.state = STILL
                self.timer = 0
                self.check_timer = 0
                self.change_image(2)
                
        elif self.state == STILL:
            if self.check_timer >= self.config.check_interval:
                self.check_timer = 0

                # Check if the agent is still on a site and update site_id
                on, site_pos = self.on_site()
                if on:
                    self.site_id = self.get_site_id(site_pos)

                lambda_leave = self.config.p_leave_base * math.exp(-self.config.beta * n_neighbors) # more neighbors = lower leave probability
                p_leave = 1.0 - math.exp(-lambda_leave * dt)
                
                if random.random() < p_leave:
                    self.state = LEAVING
                    self.timer = 0
                    self.change_image(0)
                    
        elif self.state == LEAVING:
            if self.timer >= self.config.t_leave:
                self.state = WANDERING
                self.timer = 0
                self.change_image(0)
                self.site_id = -1  # Reset site association when returning to wandering

    def change_position(self):
        if self.there_is_no_escape():   # border
            self.pos += self.move
        else:
        
            if self.state == STILL:
                self.move = Vector2(0, 0)

            elif self.state == JOINING:
                on, site_pos = self.on_site()

                # if the agent is not on the site but detects some site within it's radius
                if not on and site_pos is not None:
                    self.move = self.move_to_site(site_pos)  # keep steering in
                else:
                    self.move = Vector2(0, 0)

            elif self.state in [WANDERING, LEAVING]:
                angle = random.uniform(0, 2*math.pi)    # slight randomness for wandering/leaving states
                direction = Vector2(math.cos(angle), math.sin(angle))
                self.move = direction * self.config.movement_speed

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

    def get_site_id(self, site_pos: Vector2) -> int:
        for idx, (x, y, _) in enumerate(self.config.sites):
            if abs(Vector2(x, y).x - site_pos.x) < 1 and abs(Vector2(x, y).y - site_pos.y) < 1:
                return idx
        return -1


print("Running Aggregation - Experiment 1: Different Sized Sites (10 runs)")
cfg1 = AggregationConfig()
w, h = cfg1.window_size

all_data1 = []
runs = 30
for run in range(runs):
    print(f"Run {run + 1}/{runs}")
    sim1 = (
        Simulation(cfg1)
        .batch_spawn_agents(100, AggregationAgent, images=[
            "images/triangle.png",
            "images/triangle2.png",
            "images/green.png",
            "images/triangle3.png"
        ])
        .spawn_site("images/circle.png", x=w/3, y=h/2)
        .spawn_site("images/circle2.png", x=(w/3)*2, y=h/2)
        .run()
    )

    raw_data = sim1.snapshots.to_pandas()
    _frames = 0
    
    # Add site_id column based on position for still agents
    def assign_site_id(row):
        x, y = row['x'], row['y']
        for site_idx, (site_x, site_y, site_radius) in enumerate(cfg1.sites):
            if math.sqrt((x - site_x)**2 + (y - site_y)**2) <= site_radius * 1.1:  # 10% tolerance
                return site_idx
        return -1
    
    raw_data['site_id'] = raw_data.apply(assign_site_id, axis=1)
    
    # Group by frame and state/site
    grouped = (
        raw_data
        .groupby(['frame'])
        .agg(
            wandering=('image_index', lambda x: (x == 0).sum()),
            joining=('image_index', lambda x: (x == 1).sum()),
            leaving=('image_index', lambda x: (x == 3).sum()),
            still_siteA=('image_index', lambda x: ((x == 2) & (raw_data.loc[x.index, 'site_id'] == 0)).sum()),
            still_siteB=('image_index', lambda x: ((x == 2) & (raw_data.loc[x.index, 'site_id'] == 1)).sum())
        )
        .reset_index()
    )
    grouped['run'] = run
    data_run = pl.from_pandas(grouped)
    all_data1.append(data_run)

combined_data1 = pl.concat(all_data1)

stats1 = (
    combined_data1
    .group_by(["frame"])
    .agg([
        pl.col("wandering").alias("wandering"),
        pl.col("joining").alias("joining"),
        pl.col("leaving").alias("leaving"),
        pl.col("still_siteA").alias("stillA"),
        pl.col("still_siteB").alias("stillB")
    ])
    .sort(["frame"])
)

stats1_pd = stats1.to_pandas()

fig, ax = plt.subplots(figsize=(12, 8))

states = [
    ("Wandering", "wandering", "blue"),
    ("Joining", "joining", "orange"),
    ("Still at Small Site (A)", "stillA", "green"),
    ("Still at Large Site (B)", "stillB", "purple"),
    ("Leaving", "leaving", "red")
]
'''
for label, mean_col, color in states:
    ax.plot(stats1_pd['frame'], stats1_pd[mean_col], 
            color=color, label=label, linewidth=2)
    ax.fill_between(stats1_pd['frame'],
                    stats1_pd[mean_col],
                    stats1_pd[mean_col],
                    color=color, alpha=0.2)

ax.set_xlabel('Frame')
ax.set_ylabel('Number of Agents')
ax.set_title('Agent States Over Time (Mean ± Std Dev) for 30 Runs')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiment1_different_sizes.png", dpi=300, bbox_inches='tight')
print("Combined plot saved to experiment1_different_sizes.png")'''

# Save statistics correctly with both site A and B
with open("experiment1_different_sizes.txt", "w") as f:
    f.write("frame,wandering,joining,leaving,stillA,stillB\n")
    for row in stats1.rows():
        f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")

print("Statistics saved to experiment1_different_sizes.txt")