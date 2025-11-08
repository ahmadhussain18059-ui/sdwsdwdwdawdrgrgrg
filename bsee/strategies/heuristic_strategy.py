"""
Heuristic strategy for BSEE.
"""

import random
import yaml
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict, deque
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State


class HeuristicStrategy(BaseStrategy):
    """Multi-criteria heuristic strategy for intelligent operation selection."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize heuristic strategy."""
        super().__init__(config)
        self.load_config()
        self.operations_registry = None  # Will be set by the pipeline

        # Configuration parameters
        self.heuristic_weights = self.config.get('heuristic_weights', {
            'cost': 0.2,
            'improvement': 0.3,
            'diversity': 0.2,
            'pattern': 0.2,
            'history': 0.1
        })
        self.heuristic_window = self.config.get('heuristic_window', 10)
        self.diversity_threshold = self.config.get('diversity_threshold', 0.1)
        self.pattern_sensitivity = self.config.get('pattern_sensitivity', 0.5)

        # History tracking
        self.operation_history = deque(maxlen=self.heuristic_window)
        self.operation_success_count = defaultdict(int)
        self.operation_usage_count = defaultdict(int)
        self.pattern_memory = {}

        # Operation categories and their characteristics
        self.operation_categories = {
            'transform': ['dct_transform', 'dwt_transform', 'fft_transform', 'burrows_wheeler'],
            'compression': ['huffman_encode', 'run_length_encode', 'arithmetic_encode', 'lz77_encode'],
            'bitwise': ['xor_constant', 'toggle_bit', 'rotate_left', 'rotate_right'],
            'reordering': ['shuffle_bytes', 'reverse_bytes', 'move_to_front'],
            'delta': ['delta_encode', 'zigzag_encode']
        }

    def load_config(self) -> None:
        """Load configuration from YAML file with fallbacks."""
        config_name = "strategy_heuristic.yaml"
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
        """Create default heuristic strategy configuration file."""
        config_name = "strategy_heuristic.yaml"
        config_path = Path("config/strategies") / config_name

        default_config = {
            'name': 'Heuristic Strategy',
            'description': 'Multi-criteria heuristic evaluation for intelligent operation selection',
            'heuristic_weights': {
                'cost': 0.2,        # Computational cost heuristic
                'improvement': 0.3,  # Expected improvement heuristic
                'diversity': 0.2,    # Operation diversity heuristic
                'pattern': 0.2,      # Data pattern heuristic
                'history': 0.1       # Historical performance heuristic
            },
            'heuristic_window': 10,
            'diversity_threshold': 0.1,
            'pattern_sensitivity': 0.5,
            'adaptation_rate': 0.1
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
        """Propose operation using multi-criteria heuristic evaluation."""
        if self.operations_registry is None:
            return self._fallback_proposal(current_state)

        try:
            # Get available operations
            available_operations = list(self.operations_registry.operations.keys())

            if not available_operations:
                return self._fallback_proposal(current_state)

            # Evaluate all operations using heuristics
            operation_scores = []
            for operation in available_operations:
                heuristic_score = self.evaluate_operation_heuristic(operation, current_state)
                operation_scores.append((operation, heuristic_score))

            # Select best operation
            best_operation = max(operation_scores, key=lambda x: x[1])

            # Store for statistics tracking
            self.last_proposed_operation = best_operation[0]

            # Generate parameters for the selected operation
            params = self._generate_operation_params(best_operation[0])

            return best_operation[0], params

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

    def evaluate_operation_heuristic(self, operation: str, current_state: State) -> float:
        """Score operation using multiple heuristic criteria."""
        heuristics = {}

        # Cost heuristic: prefer less computationally expensive operations
        heuristics['cost'] = self.evaluate_cost_heuristic(operation)

        # Improvement heuristic: expected score improvement
        heuristics['improvement'] = self.evaluate_improvement_heuristic(operation, current_state)

        # Diversity heuristic: promote operations different from recent history
        heuristics['diversity'] = self.evaluate_diversity_heuristic(operation)

        # Pattern heuristic: operations effective on current data patterns
        heuristics['pattern'] = self.evaluate_pattern_heuristic(operation, current_state)

        # History heuristic: operations that performed well in past iterations
        heuristics['history'] = self.evaluate_history_heuristic(operation)

        # Combine heuristics using weights
        combined_score = self.combine_heuristics(heuristics)

        return combined_score

    def evaluate_cost_heuristic(self, operation: str) -> float:
        """Estimate computational cost of operation."""
        # Simplified cost estimation based on operation category
        if operation in ['dct_transform', 'dwt_transform', 'fft_transform']:
            return 0.3  # High cost mathematical operations
        elif operation in ['huffman_encode', 'arithmetic_encode', 'lz77_encode']:
            return 0.5  # Medium cost compression operations
        elif operation in ['xor_constant', 'toggle_bit', 'rotate_left', 'rotate_right']:
            return 1.0  # Low cost bitwise operations
        elif operation in ['shuffle_bytes', 'reverse_bytes', 'move_to_front']:
            return 0.8  # Medium cost reordering operations
        else:
            return 0.6  # Default medium cost

    def evaluate_improvement_heuristic(self, operation: str, current_state: State) -> float:
        """Expected score improvement based on operation type."""
        # Base improvement scores by operation type
        base_improvements = {
            'dct_transform': 0.7,    # Good for frequency patterns
            'huffman_encode': 0.8,   # Good for repetitive data
            'xor_constant': 0.5,     # Moderate improvement potential
            'shuffle_bytes': 0.6,    # Good for structured data
            'move_to_front': 0.7,    # Good for localized patterns
            'burrows_wheeler': 0.8,  # Good for repeated sequences
        }

        base_score = base_improvements.get(operation, 0.5)

        # Adjust based on current state generation (avoid over-processing)
        generation_factor = max(0.1, 1.0 - (current_state.generation * 0.01))

        return base_score * generation_factor

    def evaluate_diversity_heuristic(self, operation: str) -> float:
        """Promote operations different from recent history."""
        if not self.operation_history:
            return 1.0  # No history, full diversity bonus

        # Count recent occurrences of this operation
        recent_count = sum(1 for op, _ in self.operation_history if op == operation)
        diversity_score = max(0.1, 1.0 - (recent_count / len(self.operation_history)))

        return diversity_score

    def evaluate_pattern_heuristic(self, operation: str, current_state: State) -> float:
        """Operations effective on current data patterns."""
        # Simplified pattern detection based on data characteristics
        data_size = len(current_state.binary_data)
        data_entropy = self._calculate_entropy(current_state.binary_data)

        # Pattern matching scores
        pattern_scores = {}

        # High entropy data: compression operations
        if data_entropy > 7.0:
            pattern_scores.update({
                'huffman_encode': 0.9,
                'arithmetic_encode': 0.9,
                'lz77_encode': 0.8
            })

        # Low entropy data: transformation operations
        elif data_entropy < 4.0:
            pattern_scores.update({
                'dct_transform': 0.8,
                'fft_transform': 0.8,
                'shuffle_bytes': 0.7
            })

        # Medium entropy: mixed approach
        else:
            pattern_scores.update({
                'xor_constant': 0.7,
                'move_to_front': 0.7,
                'burrows_wheeler': 0.8
            })

        # Size-based preferences
        if data_size > 10000:
            pattern_scores['run_length_encode'] = 0.8
        elif data_size < 1000:
            pattern_scores['toggle_bit'] = 0.6

        return pattern_scores.get(operation, 0.5) * self.pattern_sensitivity + 0.5

    def evaluate_history_heuristic(self, operation: str) -> float:
        """Operations that performed well in past iterations."""
        if self.operation_usage_count[operation] == 0:
            return 0.5  # No history, neutral score

        # Success rate
        success_rate = (self.operation_success_count[operation] /
                       self.operation_usage_count[operation])

        # Add some exploration factor
        exploration_bonus = min(0.2, 1.0 / math.sqrt(1 + self.operation_usage_count[operation]))

        return success_rate * 0.8 + exploration_bonus * 0.2

    def combine_heuristics(self, heuristics: Dict[str, float]) -> float:
        """Weighted combination of different heuristic scores."""
        combined_score = 0.0
        total_weight = 0.0

        for heuristic_name, score in heuristics.items():
            weight = self.heuristic_weights.get(heuristic_name, 0.2)
            combined_score += score * weight
            total_weight += weight

        if total_weight > 0:
            combined_score /= total_weight

        return combined_score

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of binary data."""
        if not data:
            return 0.0

        # Count byte frequencies
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1

        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        for count in freq.values():
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)

        return entropy

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

    def select_best_heuristic(self, operation_scores: List[Tuple[str, float]]) -> Tuple[str, Dict[str, Any]]:
        """Choose operation based on combined heuristic score."""
        if not operation_scores:
            return self._fallback_proposal(None)

        # Sort by heuristic score (descending)
        operation_scores.sort(key=lambda x: x[1], reverse=True)

        # Select best operation
        best_operation = operation_scores[0][0]
        params = self._generate_operation_params(best_operation)

        return best_operation, params

    def update_statistics(self, operation: str, success: bool) -> None:
        """Update operation history and success statistics."""
        self.operation_history.append((operation, success))
        self.operation_usage_count[operation] += 1

        if success:
            self.operation_success_count[operation] += 1

        # Adapt heuristic weights based on performance
        self._adapt_heuristic_weights()

    def _adapt_heuristic_weights(self) -> None:
        """Adapt heuristic weights based on recent performance."""
        if len(self.operation_history) < self.heuristic_window:
            return

        # Calculate recent success rates
        recent_success_by_category = defaultdict(list)

        for operation, success in self.operation_history:
            category = self._get_operation_category(operation)
            recent_success_by_category[category].append(success)

        # Simple adaptation: increase weight for successful heuristic types
        adaptation_rate = self.config.get('adaptation_rate', 0.1)

        for category, successes in recent_success_by_category.items():
            if len(successes) >= 3:  # Only adapt with sufficient data
                success_rate = sum(successes) / len(successes)

                if success_rate > 0.7:
                    # Increase weights for heuristics that favor this category
                    self._increase_category_weights(category, adaptation_rate)
                elif success_rate < 0.3:
                    # Decrease weights for unsuccessful categories
                    self._decrease_category_weights(category, adaptation_rate)

    def _get_operation_category(self, operation: str) -> str:
        """Get the category of an operation."""
        for category, operations in self.operation_categories.items():
            if operation in operations:
                return category
        return 'other'

    def _increase_category_weights(self, category: str, rate: float) -> None:
        """Increase weights for heuristics that favor a category."""
        # Simplified: adjust based on category characteristics
        if category == 'compression':
            self.heuristic_weights['improvement'] = min(1.0, self.heuristic_weights['improvement'] + rate)
        elif category == 'transform':
            self.heuristic_weights['pattern'] = min(1.0, self.heuristic_weights['pattern'] + rate)
        elif category == 'bitwise':
            self.heuristic_weights['cost'] = min(1.0, self.heuristic_weights['cost'] + rate)

    def _decrease_category_weights(self, category: str, rate: float) -> None:
        """Decrease weights for heuristics that favor a category."""
        if category == 'compression':
            self.heuristic_weights['improvement'] = max(0.1, self.heuristic_weights['improvement'] - rate)
        elif category == 'transform':
            self.heuristic_weights['pattern'] = max(0.1, self.heuristic_weights['pattern'] - rate)
        elif category == 'bitwise':
            self.heuristic_weights['cost'] = max(0.1, self.heuristic_weights['cost'] - rate)

    def accept(self, new_state: State) -> bool:
        """Accept based on heuristic evaluation."""
        # Update statistics based on acceptance
        if hasattr(self, 'last_proposed_operation'):
            self.update_statistics(self.last_proposed_operation, new_state.score > self.best_score)

        return new_state.score > self.best_score

    def get_heuristic_info(self) -> Dict[str, Any]:
        """Get information about heuristic strategy state."""
        return {
            'heuristic_weights': self.heuristic_weights.copy(),
            'operation_success_rates': {
                op: (self.operation_success_count[op] / max(1, self.operation_usage_count[op]))
                for op in self.operation_usage_count
            },
            'recent_operations': list(self.operation_history),
            'pattern_sensitivity': self.pattern_sensitivity
        }

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.operation_history.clear()
        self.operation_success_count.clear()
        self.operation_usage_count.clear()
        self.pattern_memory.clear()