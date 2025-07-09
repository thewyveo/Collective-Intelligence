from dataclasses import dataclass
import platform, vi, jinja2, markupsafe, polars
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import seaborn as sns
import polars as pl
import math
from tqdm import tqdm
import random

iters = 0


@dataclass
class FlockingConfig(
    Config):  #  fastest convergence params for obstacles were all 0.5 (based on observations) therefore the
    #  obstacle script also has each parameter weight set to 0.5
    alignment_weight: float = 0.5
    cohesion_weight: float = 0.5
    separation_weight: float = 0.5
    delta_time: float = 3
    mass: int = 20
    window_size: tuple[int, int] = (750, 750)
    record_snapshots: bool = True

    def weights(self) -> tuple[float, float, float]:
        '''
        Return the weights for alignment, cohesion, and separation.
        :return: (alignment_weight, cohesion_weight, separation_weight): weights of each behavior
        '''
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight)


class FlockingAgent(Agent):

    def update(self) -> None:
        '''
        Update the agent's state (image) based on neighbors
        '''
        if any(self.in_proximity_accuracy()):
            self.change_image(1)
        else:
            self.change_image(0)

    def change_position(self) -> None:
        '''
        Change the agent's position based on neighbors' positions/velocities
        and apply flocking behavior (alignment, cohesion, separation)
        '''

        self.there_is_no_escape()

        neighbors = list(self.in_proximity_accuracy())  #  get neighbors in proximity

        global iters  # frame/iteration counter
        if self.id == 0:  # only one agent does the counting
            iters += 1
            #if iters == 2500 or iters == 5000:
            #    print('iteration #', iters)
        

        if neighbors:  # IF THERE ARE NEIGHBORS NEARBY

            alignment = Vector2(0, 0)  #  initialize alignment/cohesion/separation vectors
            cohesion = Vector2(0, 0)
            separation = Vector2(0, 0)

            self_pos = self.pos  #  get self postiion & velocity
            self_vel = self.move
            n = len(neighbors)  #  nr. neighbors

            for other, _ in neighbors:  #  iterate through neighbors & compute their positions/velocities
                other_pos = other.pos
                other_vel = other.move

                alignment += other_vel  #  add velocities of each neighbor to alignment (to be divided by n later for average)
                cohesion += other_pos  #  add positions of each neighbor to cohesion (to be divided by n later for average)
                separation += (
                            self_pos - other_pos)  #  add the difference between self position and neighbor position to separation (to be divided by n later for average)
            #  after the loop, alignment, cohesion, and separation are now the sum of all neighbors'
            # velocities/positions/differences, and they're ready to be averaged.

            alignment = ((
                                     alignment / n) - self_vel)  #  Vn - Vboid where Vn is the avg. velocities of the neighbors (alignment/n)
            cohesion = ((
                                    cohesion / n) - self_pos)  #  fc - Vboid where fc = Xn - Xboid where Xn is the avg. positions of neighbors (cohesion/n)
            separation = (
                        separation / n)  #  the mean of the differences between boid's position and neighbors' positions

            # normalizing (so one rule doesn't dominate the others)
            if alignment.length() > 0:  # avoid normalizing zero vectors (NaN values)
                alignment = alignment.normalize()  #  built-in method
            if cohesion.length() > 0:
                cohesion = cohesion.normalize()
            if separation.length() > 0:
                separation = separation.normalize()

            alignment_weight, cohesion_weight, separation_weight = self.config.weights()
            total_force = (
                                  alignment * alignment_weight
                                  + cohesion * cohesion_weight
                                  + separation * separation_weight
                          ) / self.config.mass  #  "ftotal = (alpha*alignment + beta*cohesion + gamma*separation) / mass" from Assignment_0.pdf

            self.move += total_force * self.config.delta_time  #  move update


        else:  #  NO NEIGHBORS NEARBY
            random_angle = random.uniform(0, 2 * 3.14159)  # random walk if no neighbors (random angle rotation)
            random_vector = Vector2(1, 0).rotate_rad(random_angle)
            self.move += random_vector * 0.2  # scale down the randomness
            # (otherwise agents would 'jump' or move too fast when there are no neighbors)

        #  setting max speed boundary
        max_speed = 3
        if self.move.length() > max_speed:
            self.move = self.move.normalize() * max_speed

        # updating position
        self.pos += self.move * self.config.delta_time

        self.save_data("vx", self.move.x)
        self.save_data("vy", self.move.y)

'''
def run_flocking_neuroevolution(
    generations=10,
    population_size=20,
    mutation_rate=0.1,
    crossover_rate=0.5,
    elite_count=2,
    tournament_k=3,
    early_stopping=True,
    patience=5,
    min_delta=1e-4
    ):
        param_bounds = {
            "alignment_weight": (0.0, 1.0),
            "cohesion_weight": (0.0, 1.0),
            "separation_weight": (0.0, 1.0),
        }

        def random_genome():
            return_statement = tuple(random.uniform(*param_bounds[k]) for k in param_bounds)
            print(return_statement)
            return return_statement

        def mutate(genome):
            return_statement = tuple(
                min(max(g + random.gauss(0, mutation_rate * (param_bounds[k][1] - param_bounds[k][0])), param_bounds[k][0]), param_bounds[k][1])
                for g, k in zip(genome, param_bounds)
            )
            print(return_statement)
            return return_statement

        def crossover(p1, p2):
            return_statement = tuple(p1[i] if random.random() < crossover_rate else p2[i] for i in range(len(p1)))
            print(return_statement)
            return return_statement

        def tournament_selection(pop, scores):
            best = None
            for _ in range(tournament_k):
                i = random.randint(0, len(pop) - 1)
                if best is None or scores[i] > scores[best]:
                    best = i
            print(best) if best is not None else print("0.0")
            return pop[best]
        
        def evaluate_fitness(genome):
            try:
                alignment_weight, cohesion_weight, separation_weight = genome

                cfg = FlockingConfig(
                    alignment_weight=alignment_weight,
                    cohesion_weight=cohesion_weight,
                    separation_weight=separation_weight,
                    radius=50,
                    duration=1000,
                    image_rotation=True
                )

                sim = (
                    HeadlessSimulation(cfg)
                    .batch_spawn_agents(100, FlockingAgent, images=["images/triangle.png", "images/green.png"])
                    .run()
                )

                final_frame = sim.snapshots["frame"].max()
                data = sim.snapshots.filter(pl.col("frame") == final_frame)

                # Positions and movement vectors
                positions = list(zip(data["x"].to_list(), data["y"].to_list()))
                velocities = list(zip(data["vx"].to_list(), data["vy"].to_list()))
                n = len(positions)

                # ---- Cohesion: average pairwise inverse distance
                total_inv_dist = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist > 0:
                            total_inv_dist += 1 / dist

                max_pairs = n * (n - 1) / 2
                cohesion_score = total_inv_dist / max_pairs
                print(f"cohesion score: {cohesion_score:.4f}")

                # ---- Alignment: average cosine similarity of movement vectors
                ref_vector = Vector2(*velocities[0])
                total_alignment = 0
                for vx, vy in velocities[1:]:
                    v = Vector2(vx, vy)
                    if ref_vector.length() > 0 and v.length() > 0:
                        cos_sim = ref_vector.normalize().dot(v.normalize())
                        total_alignment += cos_sim
                alignment_score = total_alignment / (n - 1)
                print(f"alignment score: {alignment_score:.4f}")

                # ---- Cluster penalty
                from collections import defaultdict, deque
                adj = defaultdict(list)
                radius = cfg.radius
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist <= radius:
                            adj[i].append(j)
                            adj[j].append(i)

                visited = set()
                num_clusters = 0
                for i in range(n):
                    if i not in visited:
                        q = deque([i])
                        while q:
                            u = q.popleft()
                            if u in visited:
                                continue
                            visited.add(u)
                            q.extend(adj[u])
                        num_clusters += 1

                cluster_penalty = 0.1 * (num_clusters - 1)
                print("cluster penalty:", cluster_penalty)

                # ---- Separation: Penalize being too close
                too_close_penalty = 0
                min_dist = 10  # or 15, depending on visual density

                for i in range(n):
                    for j in range(i + 1, n):
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < min_dist:
                            too_close_penalty += (min_dist - dist) / min_dist  # normalized penalty
                print("separation penalty:", too_close_penalty)

                # ---- Final fitness
                fitness = (
                    0.6 * cohesion_score + 
                    0.4 * alignment_score - 
                    cluster_penalty - 
                    0.2 * too_close_penalty / max_pairs
                )
                print(f"fitness: {max(0, fitness):.4f}")
                return max(0.0, min(1.0, fitness))

            except Exception as e:
                print(f"[⚠️] Fitness error: {e}")
                return 0.0

        population = []
        for _ in tqdm(range(population_size), desc="initializing population"):
            population.append(random_genome())
        fitness_scores = []
        for genome in tqdm(range(len(population)), desc="evaluating fitness of initial population"):
            fitness_scores.append(evaluate_fitness(population[genome]))
        best_ever = max(fitness_scores)
        best_genome = population[fitness_scores.index(best_ever)]
        patience_counter = 0

        for gen in tqdm(range(generations), desc="main evolution loop", lock_args=(True,)):
            print(f"\n🔁 Iteration {gen + 1}/{generations}")

            sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
            new_population = [population[i] for i in sorted_indices[:elite_count]]

            while len(new_population) < population_size:
                p1 = tournament_selection(population, fitness_scores)
                p2 = tournament_selection(population, fitness_scores)
                child = crossover(p1, p2)
                child = mutate(child)
                new_population.append(child)

            population = new_population
            print("new population created")
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
    best_genome, best_score = run_flocking_neuroevolution()
    print("\n🏆 Best Parameters:")
    for k, v in zip(["alignment", "cohesion", "separation"], best_genome):
        print(f"{k}_weight: {v:.4f}")
    print(f"→ Final fitness: {best_score:.4f}")
'''


    #  config setup
sim = (
    Simulation(
        FlockingConfig(
            image_rotation=True,
            movement_speed=1,
            radius=50,
            duration=10000  #  max 10k frames/iterations
        )
    )
    .batch_spawn_agents(100, FlockingAgent, images=["images/triangle.png", "images/green.png"])
    .run()
    .snapshots
    .group_by(["frame", "image_index"])
    .agg(pl.count("id").alias("agents"))
    .sort(["frame", "image_index"])
)

# plotting
print(sim)
plot = sns.relplot(x=sim["frame"], y=sim["agents"], hue=sim["image_index"], kind="line")
plot.savefig("agents_regular_flocking.png", dpi=300)