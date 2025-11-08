"""
Simulated annealing strategy for BSEE.
"""

import random
import math
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State


class AnnealingStrategy(BaseStrategy):
    """Simulated annealing strategy with temperature-aware operation selection."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize annealing strategy."""
        super().__init__(config)
        self.load_config()
        self.operations_registry = None  # Will be set by the pipeline

        # Configuration parameters
        self.initial_temperature = self.config.get('initial_temperature', 100.0)
        self.min_temperature = self.config.get('min_temperature', 0.1)
        self.cooling_schedule = self.config.get('cooling_schedule', 'geometric')
        self.cooling_rate = self.config.get('cooling_rate', 0.95)
        self.current_temperature = self.initial_temperature

        # Temperature thresholds for operation selection
        self.high_temp_threshold = self.config.get('high_temp_threshold', 50.0)
        self.low_temp_threshold = self.config.get('low_temp_threshold', 5.0)

        # Operation categories for temperature-based selection
        self.high_temp_operations = self.config.get('high_temp_operations', [
            'shuffle_bytes', 'reverse_bytes', 'move_to_front', 'burrows_wheeler'
        ])
        self.low_temp_operations = self.config.get('low_temp_operations', [
            'xor_constant', 'toggle_bit', 'rotate_left', 'rotate_right'
        ])

        # Acceptance criteria
        self.use_boltzmann = self.config.get('use_boltzmann', True)

    def load_config(self) -> None:
        """Load configuration from YAML file with fallbacks."""
        config_name = "strategy_annealing.yaml"
        config_path = Path("config/strategies") / config_name

        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Extract parameters from nested structure
            if 'parameters' in yaml_config:
                params = yaml_config['parameters']
                self.config.update(params)

            if 'temperature_schedule' in yaml_config:
                temp_config = yaml_config['temperature_schedule']
                self.config.update(temp_config)

            if 'acceptance' in yaml_config:
                accept_config = yaml_config['acceptance']
                self.config.update(accept_config)

            if 'operations' in yaml_config:
                ops_config = yaml_config['operations']
                self.config.update(ops_config)

        except FileNotFoundError:
            # Use default values if config file missing
            pass
        except yaml.YAMLError as e:
            # Log error and use defaults
            pass

    def set_operations_registry(self, operations_registry) -> None:
        """Set the operations registry for accessing available operations."""
        self.operations_registry = operations_registry

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using temperature-aware selection."""
        if self.operations_registry is None:
            return self._fallback_proposal(current_state)

        try:
            # Generate move based on current temperature
            operation, params = self.generate_move()

            return operation, params

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

    def generate_move(self) -> Tuple[str, Dict[str, Any]]:
        """Generate move based on current temperature."""
        # Determine operation category based on temperature
        if self.current_temperature > self.high_temp_threshold:
            # High temperature: exploratory operations
            candidate_operations = self._get_exploratory_operations()
        elif self.current_temperature < self.low_temp_threshold:
            # Low temperature: exploitative operations
            candidate_operations = self._get_exploitative_operations()
        else:
            # Medium temperature: mix of both
            exploratory = self._get_exploratory_operations()
            exploitative = self._get_exploitative_operations()
            candidate_operations = exploratory + exploitative

        if not candidate_operations:
            return self._fallback_proposal(None)

        # Select operation from candidates
        operation = random.choice(candidate_operations)
        params = self._generate_operation_params(operation)

        return operation, params

    def _get_exploratory_operations(self) -> List[str]:
        """Get list of exploratory operations available."""
        if self.operations_registry:
            available_ops = list(self.operations_registry.operations.keys())
            # Filter for high-temperature operations that are available
            exploratory = [op for op in self.high_temp_operations if op in available_ops]
            if exploratory:
                return exploratory
            # Fallback to any available operations
            return available_ops[:5]
        return self.high_temp_operations

    def _get_exploitative_operations(self) -> List[str]:
        """Get list of exploitative operations available."""
        if self.operations_registry:
            available_ops = list(self.operations_registry.operations.keys())
            # Filter for low-temperature operations that are available
            exploitative = [op for op in self.low_temp_operations if op in available_ops]
            if exploitative:
                return exploitative
            # Fallback to any available operations
            return available_ops[:5]
        return self.low_temp_operations

    def _generate_operation_params(self, operation: str) -> Dict[str, Any]:
        """Generate parameters for an operation."""
        if operation == 'xor_constant':
            return {'constant': random.randint(1, 255)}
        elif operation == 'rotate_left' or operation == 'rotate_right':
            return {'shift': random.randint(1, 7)}
        elif operation == 'shuffle_bytes':
            return {'seed': random.randint(0, 10000)}
        elif operation == 'toggle_bit':
            return {'position': random.randint(0, 7)}
        else:
            return {}

    def accept(self, new_state: State, current_score: float = None) -> bool:
        """Accept based on simulated annealing criteria with Boltzmann acceptance."""
        if current_score is None:
            current_score = self.best_score

        # Always accept better states
        if new_state.score > current_score:
            return True

        # Use Boltzmann acceptance for worse states
        if self.use_boltzmann and self.current_temperature > self.min_temperature:
            delta_score = new_state.score - current_score
            acceptance_probability = math.exp(delta_score / self.current_temperature)

            if random.random() < acceptance_probability:
                return True

        # Cool down temperature
        self.update_temperature()

        return False

    def update_temperature(self) -> None:
        """Update temperature based on cooling schedule."""
        if self.cooling_schedule == 'geometric':
            self.current_temperature *= self.cooling_rate
        elif self.cooling_schedule == 'linear':
            self.current_temperature -= self.cooling_rate
        elif self.cooling_schedule == 'logarithmic':
            # Simplified logarithmic cooling
            self.current_temperature = self.initial_temperature / (1 + math.log(1 + self.iteration_count))
        else:
            # Default to geometric
            self.current_temperature *= self.cooling_rate

        # Ensure temperature doesn't go below minimum
        self.current_temperature = max(self.current_temperature, self.min_temperature)

    def get_temperature_info(self) -> Dict[str, Any]:
        """Get information about current temperature state."""
        return {
            'current_temperature': self.current_temperature,
            'initial_temperature': self.initial_temperature,
            'min_temperature': self.min_temperature,
            'cooling_ratio': self.current_temperature / self.initial_temperature,
            'temperature_phase': self._get_temperature_phase()
        }

    def _get_temperature_phase(self) -> str:
        """Get current temperature phase."""
        if self.current_temperature > self.high_temp_threshold:
            return 'exploration'
        elif self.current_temperature < self.low_temp_threshold:
            return 'exploitation'
        else:
            return 'transition'

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.current_temperature = self.initial_temperature