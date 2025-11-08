"""
Genetic algorithm strategy for BSEE.
"""

import random
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State


class Individual:
    """Represents an individual in the genetic algorithm population."""

    def __init__(self, operations: Optional[List[Tuple[str, Dict[str, Any]]]] = None):
        self.operations = operations or []  # List of (operation_name, params) tuples
        self.fitness = float('-inf')
        self.age = 0

    def copy(self) -> 'Individual':
        """Create a copy of this individual."""
        return Individual(self.operations.copy())

    def get_length(self) -> int:
        """Get the length of the individual's operation sequence."""
        return len(self.operations)

    def add_operation(self, operation: Tuple[str, Dict[str, Any]]) -> None:
        """Add an operation to the individual."""
        self.operations.append(operation)

    def get_last_operation(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get the last operation in the individual."""
        if self.operations:
            return self.operations[-1]
        return None


class Population:
    """Population of individuals for genetic algorithm."""

    def __init__(self, size: int, operations_registry):
        self.individuals: List[Individual] = []
        self.size = size
        self.operations_registry = operations_registry
        self.generation = 0
        self.best_individual = None
        self.average_fitness = 0.0

    def initialize_random(self, current_state: State) -> None:
        """Initialize population with random individuals."""
        self.individuals = []
        for _ in range(self.size):
            individual = Individual()

            # Random length between 1 and 10 operations
            length = random.randint(1, 10)

            for _ in range(length):
                if self.operations_registry:
                    operation = random.choice(list(self.operations_registry.operations.keys()))
                else:
                    operation = random.choice(['xor_constant', 'rotate_left', 'move_to_front', 'shuffle_bytes'])

                params = self._generate_operation_params(operation)
                individual.add_operation((operation, params))

            self.individuals.append(individual)

    def _generate_operation_params(self, operation: str) -> Dict[str, Any]:
        """Generate parameters for an operation."""
        if operation == 'xor_constant':
            return {'constant': random.randint(1, 255)}
        elif operation == 'rotate_left' or operation == 'rotate_right':
            return {'shift': random.randint(1, 7)}
        elif operation == 'shuffle_bytes':
            return {'seed': random.randint(0, 10000)}
        else:
            return {}

    def evaluate_fitness(self, current_state: State) -> None:
        """Evaluate fitness for all individuals."""
        total_fitness = 0.0
        best_fitness = float('-inf')

        for individual in self.individuals:
            # Simplified fitness evaluation based on operation sequence
            # In practice, this would apply operations and evaluate the resulting state
            fitness = self._evaluate_individual_fitness(individual, current_state)
            individual.fitness = fitness
            total_fitness += fitness

            if fitness > best_fitness:
                best_fitness = fitness
                self.best_individual = individual

        self.average_fitness = total_fitness / max(1, len(self.individuals))

    def _evaluate_individual_fitness(self, individual: Individual, current_state: State) -> float:
        """Evaluate fitness of an individual (simplified)."""
        # Simplified fitness: reward longer sequences and diverse operations
        fitness = len(individual.operations) * 0.1

        # Bonus for operation diversity
        operations_used = set(op[0] for op in individual.operations)
        fitness += len(operations_used) * 0.2

        # Add some randomness to simulate actual evaluation
        fitness += random.uniform(-1, 1)

        return fitness

    def get_best_individual(self) -> Optional[Individual]:
        """Get the best individual in the population."""
        if not self.individuals:
            return None

        best = self.individuals[0]
        for individual in self.individuals[1:]:
            if individual.fitness > best.fitness:
                best = individual
        return best

    def select_parents(self, method: str = "tournament", tournament_size: int = 3) -> Tuple[Individual, Individual]:
        """Select parents for crossover."""
        if method == "tournament":
            parent1 = self._tournament_selection(tournament_size)
            parent2 = self._tournament_selection(tournament_size)
        elif method == "roulette_wheel":
            parent1 = self._roulette_wheel_selection()
            parent2 = self._roulette_wheel_selection()
        elif method == "rank_based":
            parent1 = self._rank_based_selection()
            parent2 = self._rank_based_selection()
        else:
            # Default to random selection
            parent1 = random.choice(self.individuals)
            parent2 = random.choice(self.individuals)

        return parent1, parent2

    def _tournament_selection(self, tournament_size: int) -> Individual:
        """Tournament selection."""
        tournament = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
        return max(tournament, key=lambda ind: ind.fitness)

    def _roulette_wheel_selection(self) -> Individual:
        """Roulette wheel selection."""
        # Ensure all fitness values are positive
        min_fitness = min(ind.fitness for ind in self.individuals)
        adjusted_fitnesses = [ind.fitness - min_fitness + 0.1 for ind in self.individuals]

        total_fitness = sum(adjusted_fitnesses)
        if total_fitness == 0:
            return random.choice(self.individuals)

        selection_point = random.uniform(0, total_fitness)
        current_sum = 0.0

        for i, fitness in enumerate(adjusted_fitnesses):
            current_sum += fitness
            if current_sum >= selection_point:
                return self.individuals[i]

        return self.individuals[-1]

    def _rank_based_selection(self) -> Individual:
        """Rank-based selection."""
        # Sort individuals by fitness
        sorted_individuals = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)

        # Assign probability based on rank
        total_ranks = len(sorted_individuals) * (len(sorted_individuals) + 1) / 2
        selection_point = random.uniform(0, total_ranks)

        current_sum = 0.0
        for i, individual in enumerate(sorted_individuals):
            current_sum += (len(sorted_individuals) - i)
            if current_sum >= selection_point:
                return individual

        return sorted_individuals[-1]

    def evolve(self, config: Dict[str, Any]) -> None:
        """Evolve the population by one generation."""
        # Create new population
        new_population = []

        # Elite preservation
        elite_size = config.get('elite_size', 5)
        elite_count = min(elite_size, len(self.individuals))

        if elite_count > 0:
            # Sort by fitness and keep elite individuals
            sorted_individuals = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
            for i in range(elite_count):
                elite_individual = sorted_individuals[i].copy()
                elite_individual.age = 0
                new_population.append(elite_individual)

        # Generate offspring
        crossover_rate = config.get('crossover_rate', 0.8)
        mutation_rate = config.get('mutation_rate', 0.1)
        crossover_type = config.get('crossover_type', 'single_point')
        mutation_type = config.get('mutation_type', 'point')

        while len(new_population) < self.size:
            # Select parents
            parent1, parent2 = self.select_parents(config.get('selection_method', 'tournament'))

            # Crossover
            if random.random() < crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2, crossover_type)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()

            # Mutation
            offspring1 = self._mutate(offspring1, mutation_rate, mutation_type)
            offspring2 = self._mutate(offspring2, mutation_rate, mutation_type)

            # Add to new population
            new_population.extend([offspring1, offspring2])

        # Trim to exact size
        self.individuals = new_population[:self.size]

        # Age individuals
        for individual in self.individuals:
            individual.age += 1

        self.generation += 1

    def _crossover(self, parent1: Individual, parent2: Individual, crossover_type: str) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if crossover_type == "single_point":
            return self._single_point_crossover(parent1, parent2)
        elif crossover_type == "two_point":
            return self._two_point_crossover(parent1, parent2)
        elif crossover_type == "uniform":
            return self._uniform_crossover(parent1, parent2)
        else:
            # Default to single point
            return self._single_point_crossover(parent1, parent2)

    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        # Choose crossover point
        max_point = min(len(parent1.operations), len(parent2.operations))
        if max_point == 0:
            return parent1.copy(), parent2.copy()

        crossover_point = random.randint(1, max_point)

        # Create offspring
        offspring1_ops = parent1.operations[:crossover_point] + parent2.operations[crossover_point:]
        offspring2_ops = parent2.operations[:crossover_point] + parent1.operations[crossover_point:]

        return Individual(offspring1_ops), Individual(offspring2_ops)

    def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two-point crossover."""
        max_point = min(len(parent1.operations), len(parent2.operations))
        if max_point < 2:
            return self._single_point_crossover(parent1, parent2)

        point1 = random.randint(1, max_point - 1)
        point2 = random.randint(point1 + 1, max_point)

        # Create offspring by swapping middle segment
        offspring1_ops = (parent1.operations[:point1] +
                         parent2.operations[point1:point2] +
                         parent1.operations[point2:])
        offspring2_ops = (parent2.operations[:point1] +
                         parent1.operations[point1:point2] +
                         parent2.operations[point2:])

        return Individual(offspring1_ops), Individual(offspring2_ops)

    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        max_length = max(len(parent1.operations), len(parent2.operations))

        offspring1_ops = []
        offspring2_ops = []

        for i in range(max_length):
            if i < len(parent1.operations) and i < len(parent2.operations):
                if random.random() < 0.5:
                    offspring1_ops.append(parent1.operations[i])
                    offspring2_ops.append(parent2.operations[i])
                else:
                    offspring1_ops.append(parent2.operations[i])
                    offspring2_ops.append(parent1.operations[i])
            elif i < len(parent1.operations):
                offspring1_ops.append(parent1.operations[i])
                offspring2_ops.append(parent1.operations[i])
            else:
                offspring1_ops.append(parent2.operations[i])
                offspring2_ops.append(parent2.operations[i])

        return Individual(offspring1_ops), Individual(offspring2_ops)

    def _mutate(self, individual: Individual, mutation_rate: float, mutation_type: str) -> Individual:
        """Mutate an individual."""
        mutated = individual.copy()

        for i in range(len(mutated.operations)):
            if random.random() < mutation_rate:
                if mutation_type == "point":
                    mutated.operations[i] = self._point_mutation(mutated.operations[i])
                elif mutation_type == "insert":
                    mutated.operations.insert(i, self._generate_random_operation())
                elif mutation_type == "delete" and len(mutated.operations) > 1:
                    del mutated.operations[i]
                elif mutation_type == "swap" and i < len(mutated.operations) - 1:
                    mutated.operations[i], mutated.operations[i + 1] = mutated.operations[i + 1], mutated.operations[i]

        return mutated

    def _point_mutation(self, operation: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Point mutation: replace operation with different one."""
        if self.operations_registry:
            new_operation = random.choice(list(self.operations_registry.operations.keys()))
        else:
            new_operation = random.choice(['xor_constant', 'rotate_left', 'move_to_front', 'shuffle_bytes'])

        return (new_operation, self._generate_operation_params(new_operation))

    def _generate_random_operation(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a random operation."""
        if self.operations_registry:
            operation = random.choice(list(self.operations_registry.operations.keys()))
        else:
            operation = random.choice(['xor_constant', 'rotate_left', 'move_to_front', 'shuffle_bytes'])

        return (operation, self._generate_operation_params(operation))


class GeneticStrategy(BaseStrategy):
    """Genetic algorithm strategy."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize genetic strategy."""
        super().__init__(config)
        self.load_config()
        self.population = None
        self.operations_registry = None  # Will be set by the pipeline

        # Configuration parameters
        self.population_size = self.config.get('population_size', 50)
        self.crossover_rate = self.config.get('crossover_rate', 0.8)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.elite_size = self.config.get('elite_size', 5)
        self.selection_method = self.config.get('selection_method', 'tournament')
        self.crossover_type = self.config.get('crossover_type', 'single_point')
        self.mutation_type = self.config.get('mutation_type', 'point')

    def load_config(self) -> None:
        """Load configuration from YAML file with fallbacks."""
        config_name = "strategy_genetic.yaml"
        config_path = Path("config/strategies") / config_name

        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                self.config.update(yaml_config)
        except FileNotFoundError:
            # Create default config file if it doesn't exist
            self._create_default_config()
        except yaml.YAMLError as e:
            # Log error and use defaults
            pass

    def _create_default_config(self) -> None:
        """Create default genetic strategy configuration file."""
        config_name = "strategy_genetic.yaml"
        config_path = Path("config/strategies") / config_name

        default_config = {
            'name': 'Genetic Algorithm',
            'description': 'Evolutionary search with crossover and mutation',
            'population_size': 50,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'elite_size': 5,
            'selection_method': 'tournament',  # Options: 'tournament', 'roulette_wheel', 'rank_based'
            'crossover_type': 'single_point',  # Options: 'single_point', 'two_point', 'uniform'
            'mutation_type': 'point',  # Options: 'point', 'insert', 'delete', 'swap'
            'tournament_size': 3,
            'max_generations': 100
        }

        try:
            import os
            os.makedirs(config_path.parent, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        except Exception:
            pass  # Ignore if we can't create the config file

    def set_operations_registry(self, operations_registry) -> None:
        """Set the operations registry for accessing available operations."""
        self.operations_registry = operations_registry

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using genetic algorithm."""
        if self.operations_registry is None:
            return self._fallback_proposal(current_state)

        try:
            # Initialize population if needed
            if self.population is None:
                self.population = Population(self.population_size, self.operations_registry)
                self.population.initialize_random(current_state)
                self.population.evaluate_fitness(current_state)

            # Evolve population
            self.population.evolve(self.config)
            self.population.evaluate_fitness(current_state)

            # Get best individual
            best_individual = self.population.get_best_individual()
            if best_individual and best_individual.operations:
                # Return the last operation from the best individual
                return best_individual.get_last_operation() or ('xor_constant', {'constant': 1})

            # Fallback if no good individual found
            return self._fallback_proposal(current_state)

        except MemoryError:
            return self._fallback_proposal(current_state)
        except Exception as e:
            return self._fallback_proposal(current_state)

    def _fallback_proposal(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Fallback proposal using random selection."""
        if self.operations_registry:
            operations = list(self.operations_registry.operations.keys())
            if operations:
                operation = random.choice(operations)
                params = self._generate_operation_params(operation)
                return (operation, params)
        return ('xor_constant', {'constant': random.randint(1, 255)})

    def _generate_operation_params(self, operation: str) -> Dict[str, Any]:
        """Generate parameters for an operation."""
        if operation == 'xor_constant':
            return {'constant': random.randint(1, 255)}
        elif operation == 'rotate_left' or operation == 'rotate_right':
            return {'shift': random.randint(1, 7)}
        elif operation == 'shuffle_bytes':
            return {'seed': random.randint(0, 10000)}
        else:
            return {}

    def get_best_operation(self) -> Tuple[str, Dict[str, Any]]:
        """Get best operation from best individual."""
        if self.population:
            best_individual = self.population.get_best_individual()
            if best_individual and best_individual.operations:
                return best_individual.get_last_operation() or ('xor_constant', {'constant': 1})
        return ('xor_constant', {'constant': 1})

    def accept(self, new_state: State) -> bool:
        """Accept based on fitness."""
        # Could update individual fitness based on actual state evaluation
        return new_state.score > self.best_score