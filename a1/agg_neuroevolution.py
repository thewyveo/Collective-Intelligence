from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import polars as pl
import math

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
    check_interval: int = 1 * delta_time

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
            # Comment this for speed:
            # print(f"{frames-1}/5000")

        self.timer += self.config.delta_time
        self.check_timer += self.config.delta_time

        neighbors = list(self.in_proximity_accuracy())
        n_still_neighbors = sum(1 for (a, _) in neighbors if a.state == STILL)
        dt = self.config.delta_time

        random_join_chance = 0.001

        if self.state == WANDERING:
            self.change_image(WANDERING)

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
                random_angle = random.uniform(0, 2 * math.pi)
                random_vector = Vector2(1, 0).rotate_rad(random_angle)
                self.move += random_vector * 0.2
            self.pos += self.move * self.config.delta_time

def run_neuroevolution(
    generations=20,
    population_size=16,
    mutation_rate=0.1,
    crossover_rate=0.7,
    elite_count=2,
    tournament_k=3,
    early_stopping=True,
    patience=5,
    min_delta=1e-4
):
    param_bounds = {
        "p_join_base": (0.0, 0.1),
        "p_leave_base": (0.0, 0.1),
        "alpha": (0.0, 1.0),
        "beta": (0.0, 1.0),
        "movement_speed": (10.0, 50.0),
        "t_join": (1, 10),
        "t_leave": (1, 10),
        "check_interval": (0.5, 10.0)
    }

    def random_genome():
        return tuple(random.uniform(*param_bounds[k]) for k in param_bounds)

    def mutate(genome):
        return tuple(
            min(max(g + random.gauss(0, mutation_rate * (param_bounds[k][1] - param_bounds[k][0])), param_bounds[k][0]), param_bounds[k][1])
            for g, k in zip(genome, param_bounds)
        )

    def crossover(p1, p2):
        return tuple(p1[i] if random.random() < crossover_rate else p2[i] for i in range(len(p1)))

    def tournament_selection(pop, scores):
        best = None
        for _ in range(tournament_k):
            i = random.randint(0, len(pop) - 1)
            if best is None or scores[i] > scores[best]:
                best = i
        return pop[best]

    def evaluate_fitness(genome):
        try:
            (
                p_join_base, p_leave_base, alpha, beta,
                movement_speed, t_join, t_leave, check_interval
            ) = genome

            cfg = AggregationConfig(
                p_join_base=p_join_base,
                p_leave_base=p_leave_base,
                alpha=alpha,
                beta=beta,
                movement_speed=movement_speed,
                t_join=t_join,
                t_leave=t_leave,
                check_interval=check_interval
            )

            sim = (
                HeadlessSimulation(cfg)
                .batch_spawn_agents(100, AggregationAgent, images=[
                    "images/triangle.png",
                    "images/triangle5.png",
                    "images/green.png",
                    "images/triangle2.png"
                ])
                .run()
            )

            final_frame = sim.snapshots["frame"].max()
            final_data = sim.snapshots.filter(pl.col("frame") == final_frame)
            active = final_data.filter(pl.col("image_index").is_in([1, 2]))

            if active.shape[0] == 0:
                return 0.0

            positions = list(zip(active["x"].to_list(), active["y"].to_list()))
            radius = cfg.radius

            from collections import defaultdict, deque
            n = len(positions)
            adj = defaultdict(list)

            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= radius:
                        adj[i].append(j)
                        adj[j].append(i)

            visited = set()
            cluster_sizes = []

            for i in range(n):
                if i not in visited:
                    q = deque([i])
                    size = 0
                    while q:
                        u = q.popleft()
                        if u in visited:
                            continue
                        visited.add(u)
                        size += 1
                        q.extend(adj[u])
                    cluster_sizes.append(size)

            if not cluster_sizes:
                return 0.0

            largest = max(cluster_sizes)
            total_clustered = sum(cluster_sizes)
            num_clusters = len(cluster_sizes)

            # ✅ Adaptive scoring: reward growth, gently penalize fragmentation
            fitness = (largest / 100)
            fitness += 0.5 * (total_clustered / 100)
            fitness -= 0.05 * (num_clusters - 1)

            return max(0.0, min(1.0, fitness))  # Cap to [0, 1]

        except Exception as e:
            print(f"[⚠️] Eval error: {e}")
            return 0.0


    # Initialize
    population = [random_genome() for _ in range(population_size)]
    fitness_scores = [evaluate_fitness(g) for g in population]
    best_ever = max(fitness_scores)
    best_genome = population[fitness_scores.index(best_ever)]
    patience_counter = 0

    for gen in range(generations):
        print(f"\n🔁 Generation {gen + 1}")

        sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
        new_population = [population[i] for i in sorted_indices[:elite_count]]

        while len(new_population) < population_size:
            p1 = tournament_selection(population, fitness_scores)
            p2 = tournament_selection(population, fitness_scores)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        fitness_scores = [evaluate_fitness(g) for g in population]

        gen_best = max(fitness_scores)
        gen_best_genome = population[fitness_scores.index(gen_best)]

        if gen_best > best_ever + min_delta:
            best_ever = gen_best
            best_genome = gen_best_genome
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"📈 Gen best: {gen_best:.4f}")
        print(f"🏅 Best ever: {best_ever:.4f}")
        print(f"   Params: {dict(zip(param_bounds, best_genome))}")

        if early_stopping and patience_counter >= patience:
            print(f"\n🛑 Early stopping at generation {gen + 1} (no improvement in {patience} generations).")
            break

    return best_genome, best_ever


if __name__ == "__main__":
    best_params, best_score = run_neuroevolution()
    print("\n🏆 Final Best Configuration:")
    for k, v in zip([
        "p_join_base", "p_leave_base", "alpha", "beta", 
        "movement_speed", "t_join", "t_leave", "check_interval"
    ], best_params):
        print(f"{k}: {v:.4f}")
    print(f"→ Final fitness score: {best_score:.4f}")
