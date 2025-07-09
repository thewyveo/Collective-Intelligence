import numpy as np
import pygame
import random
import sys
import math
from matplotlib import pyplot as plt

# Simulation parameters
DIAMETER = 300            # Diameter of the circular agar plate
CELL_SIZE = 2              # Pixels per cell
SPECIES = 10                # Number of species (rock, paper, scissors)
EMPTY = 0                  # Value for empty space

# Toxin parameters from the paper
# Strain R (Red) is strongest, G (Green) intermediate, B (Blue) weakest
TOXIN_STRENGTH = {
    1: 1.0,  # Strain R (Red)
    2: 1.0,  # Strain G (Green)
    3: 1.0,   # Strain B (Blue)
    4: 1.0,  # Strain E (Orange)
    5: 1.0,  # Strain F (Purple)
    6: 1.0,  # Strain X (Pink)
    7: 1.0,  # Strain Y (Yellow)
    8: 1.0,  # Strain Z (Cyan)
    9: 1.0,  # Strain W (Magenta)
    10: 1.0  # Strain H (Violet)
}

# Base interaction rates
BASE_REPRODUCTION_RATE = 0.3
BASE_DEATH_RATE = 0.005
BASE_DOMINATION_RATE = 0.8  # Base rate at which one species dominates another

# Colors matching the paper
COLORS = [
    (255, 255, 255),       # Empty - white (agar)
    (255, 0, 0),           # Species 1 - red (Strain R)
    (0, 255, 0),           # Species 2 - green (Strain G)
    (0, 0, 255),            # Species 3 - blue (Strain B)
    (255, 165, 0),         # Species 4 - orange (Strain E)
    (128, 0, 128),          # Species 5 - purple (Strain F)
    (255, 192, 203),       # Species 6 - pink (Strain X)
    (255, 255, 0),         # Species 7 - yellow (Strain Y)
    (0, 255, 255),         # Species 8 - cyan (Strain Z)
    (255, 0, 255),         # Species 9 - magenta (Strain A)
    (128, 0, 0)            # Species 10 - violet (Strain B)
]

class BacterialToxinSimulation:
    def __init__(self, init_mode='grid', spacing=10):
        radius = DIAMETER // 2
        self.radius = radius
        self.center = (radius, radius)
        self.grid = np.zeros((DIAMETER, DIAMETER), dtype=int)
        self.next_grid = np.zeros_like(self.grid)
        self.population_history = []

        if init_mode == 'grid':
            self.initialize_grid_pattern(spacing)
        elif init_mode == 'vortex':
            self.initialize_vortex_pattern()
        elif init_mode == 'concentric':
            self.initialize_concentric_rings()
        elif init_mode == 'noise':
            self.initialize_noise_pattern()
        else:
            self.initialize_random()

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((DIAMETER * CELL_SIZE, DIAMETER * CELL_SIZE))
        pygame.display.set_caption("Bacterial Toxin-Mediated RPS Simulation")
        self.clock = pygame.time.Clock()

    def initialize_random(self):
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if self.is_in_circle(i, j):
                    rand_val = random.random()
                    threshold = 1.0 / (SPECIES + 1)  # +1 to leave space for EMPTY
                    assigned = False
                    for s in range(1, SPECIES + 1):
                        if rand_val < s * threshold:
                            self.grid[i, j] = s
                            assigned = True
                            break
                    if not assigned:
                        self.grid[i, j] = EMPTY
                else:
                    self.grid[i, j] = -1  # Outside agar

    def initialize_grid_pattern(self, spacing):
        strain = 1
        for i in range(0, DIAMETER, spacing):
            for j in range(0, DIAMETER, spacing):
                if self.is_in_circle(i, j):
                    self.grid[i, j] = strain
                    strain = strain % SPECIES + 1  # Cycle through 1→2→3→1...
                else:
                    self.grid[i, j] = -1

    def initialize_vortex_pattern(self, density=0.05):
        total_species = SPECIES
        angle_per_species = 2 * math.pi / total_species
        
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if not self.is_in_circle(i, j):
                    self.grid[i, j] = -1
                    continue

                dx = i - self.center[0]
                dy = j - self.center[1]
                angle = math.atan2(dy, dx)
                if angle < 0:
                    angle += 2 * math.pi

                species_index = int(angle // angle_per_species) + 1  # Species are 1-indexed
                if random.random() < density:
                    self.grid[i, j] = species_index
                else:
                    self.grid[i, j] = EMPTY

    def initialize_concentric_rings(self):
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if self.is_in_circle(i, j):
                    dx = i - self.center[0]
                    dy = j - self.center[1]
                    dist = math.hypot(dx, dy)
                    ring = int(dist / (self.radius / SPECIES))
                    self.grid[i, j] = (ring % SPECIES) + 1
                else:
                    self.grid[i, j] = -1

    def initialize_noise_pattern(self):
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if self.is_in_circle(i, j):
                    noise_val = (math.sin(i * 0.1) + math.cos(j * 0.1)) * 0.5
                    species = int((noise_val + 1) / 2 * SPECIES)
                    self.grid[i, j] = (species % SPECIES) + 1
                else:
                    self.grid[i, j] = -1
        
    def is_in_circle(self, x, y):
        # Check if coordinates are within the circular agar plate
        dx = x - self.center[0]
        dy = y - self.center[1]
        return dx*dx + dy*dy <= self.radius*self.radius
    
    def get_interaction_rate(self, attacker, defender):
        """Calculate the interaction rate based on toxin strengths"""
        # From the paper: R kills G, G kills B, B kills R
        interaction_map = {
            (1, 2): TOXIN_STRENGTH[1],  # R kills G
            (2, 3): TOXIN_STRENGTH[2],  # G kills B
            (3, 4): TOXIN_STRENGTH[3],  # B kills R
            (4, 5): TOXIN_STRENGTH[4],  # E kills F
            (5, 6): TOXIN_STRENGTH[5],  # F kills X
            (6, 7): TOXIN_STRENGTH[6],  # X kills Y
            (7, 8): TOXIN_STRENGTH[7],  # Y kills Z
            (8, 9): TOXIN_STRENGTH[8],  # Z kills W
            (9, 10): TOXIN_STRENGTH[9],  # W kills H
            (10, 1): TOXIN_STRENGTH[10],  # H kills R
        }

        # Normalize the interaction rates relative to the weakest toxin
        base_rate = interaction_map.get((attacker, defender), 0)
        return BASE_DOMINATION_RATE * (base_rate / TOXIN_STRENGTH[3])
    
    def update(self):
        self.next_grid = np.copy(self.grid)
        
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if not self.is_in_circle(i, j):
                    continue
                
                current = self.grid[i, j]
                
                # Death process
                if current != EMPTY and random.random() < BASE_DEATH_RATE:
                    self.next_grid[i, j] = EMPTY
                    continue
                
                # Reproduction and competition
                if current != EMPTY:
                    # Find a random neighbor
                    ni, nj = self.get_random_neighbor(i, j)
                    if not self.is_in_circle(ni, nj):
                        continue
                    
                    neighbor = self.grid[ni, nj]
                    
                    # Reproduction to empty space
                    if neighbor == EMPTY and random.random() < BASE_REPRODUCTION_RATE:
                        self.next_grid[ni, nj] = current
                    
                    # Toxin-mediated dominance interactions
                    elif neighbor != EMPTY and neighbor != current:
                        # Get the appropriate interaction rate based on toxin strengths
                        domination_rate = self.get_interaction_rate(current, neighbor)
                        
                        if random.random() < domination_rate:
                            self.next_grid[ni, nj] = current
        
        self.grid = self.next_grid
        
        # Record population counts
        counts = [np.sum(self.grid == i) for i in range(SPECIES + 1)]
        self.population_history.append(counts[1:])  # Exclude empty count
        
    def get_random_neighbor(self, i, j):
        # Get random neighbor (no periodic boundaries)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        di, dj = random.choice(directions)
        ni, nj = i + di, j + dj
        
        # Ensure we stay within grid bounds
        ni = max(0, min(DIAMETER - 1, ni))
        nj = max(0, min(DIAMETER - 1, nj))
        
        return ni, nj
    
    def draw(self):
        # Draw agar plate (white circle)
        pygame.draw.circle(self.screen, (255, 255, 255), 
                          (self.center[0] * CELL_SIZE, self.center[1] * CELL_SIZE), 
                          self.radius * CELL_SIZE)
        
        # Draw bacteria
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if not self.is_in_circle(i, j):
                    continue
                
                species = self.grid[i, j]
                if species != EMPTY:
                    pygame.draw.rect(self.screen, COLORS[species], 
                                   (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        pygame.display.flip()
    
    def run(self, generations):
        running = True
        for _ in range(generations):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
                
            self.update()
            self.draw()
            self.clock.tick(60)
            if _ % 100 == 0:
                print(f"Generation {_ + 1}/{generations} completed")
        
        pygame.quit()
        self.plot_population_history()
    
    def plot_population_history(self):
        history = np.array(self.population_history)
        plt.figure(figsize=(10, 6))
        plt.plot(history[:, 0], 'r-', label='Strain R (Strongest)')
        plt.plot(history[:, 1], 'g-', label='Strain G (Intermediate)')
        plt.plot(history[:, 2], 'b-', label='Strain B (Weakest)')
        plt.plot(history[:, 3], 'orange', label='Strain E (Orange)')
        plt.plot(history[:, 4], 'purple', label='Strain F (Purple)')
        plt.plot(history[:, 5], 'pink', label='Strain X (Pink)')
        plt.plot(history[:, 6], 'yellow', label='Strain Y (Yellow)')
        plt.plot(history[:, 7], 'cyan', label='Strain Z (Cyan)')
        plt.plot(history[:, 8], 'magenta', label='Strain W (Magenta)')
        plt.plot(history[:, 9], 'violet', label='Strain H (Violet)')
        plt.title('Population Dynamics with Asymmetric Toxin Strengths')
        plt.xlabel('Time Steps')
        plt.ylabel('Population Count')
        plt.legend()
        plt.grid(True)
        plt.savefig("bacteria_same.png")
        plt.close()
        print("Population history saved to bacteria_same.png")



if __name__ == "__main__":
    print("Starting simulation with toxin-mediated interactions")
    print("Toxin strength ratios (from paper):")
    print(f"Strain R (Red): {TOXIN_STRENGTH[1]} (strongest)")
    print(f"Strain G (Green): {TOXIN_STRENGTH[2]} (intermediate)")
    print(f"Strain B (Blue): {TOXIN_STRENGTH[3]} (weakest, reference)")

    # Change 'init_mode' to 'random' or 'grid'
    sim = BacterialToxinSimulation(init_mode='')
    sim.run(5000)


