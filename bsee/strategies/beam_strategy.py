"""
Beam search strategy for BSEE.
"""

import random
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State


class BeamState:
    """Represents a state in the beam search."""

    def __init__(self, state: State, operations_applied: List[Tuple[str, Dict[str, Any]]], score: float):
        self.state = state  # Current State object
        self.operations_applied = operations_applied  # History of operations
        self.score = score
        self.age = 0
        self.diversity_score = 0.0  # For diversity preservation

    def copy(self) -> 'BeamState':
        """Create a copy of this beam state."""
        return BeamState(
            state=self.state.copy(),
            operations_applied=self.operations_applied.copy(),
            score=self.score
        )

    def get_last_operation(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get the last operation applied."""
        if self.operations_applied:
            return self.operations_applied[-1]
        return None

    def __lt__(self, other: 'BeamState') -> bool:
        """Less than comparison for sorting (higher score is better)."""
        return self.score > other.score


class BeamStrategy(BaseStrategy):
    """Beam search strategy that maintains k-best candidate states."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize beam strategy."""
        super().__init__(config)
        self.load_config()
        self.beam: List[BeamState] = []
        self.operations_registry = None  # Will be set by the pipeline

        # Configuration parameters
        self.beam_width = self.config.get('beam_width', 5)
        self.max_beam_width = self.config.get('max_beam_width', 20)
        self.min_beam_width = self.config.get('min_beam_width', 2)
        self.branch_factor = self.config.get('branch_factor', 3)
        self.selection_method = self.config.get('selection_method', 'top_n')
        self.preserve_diversity = self.config.get('preserve_diversity', True)
        self.normalize_scores = self.config.get('normalize_scores', True)
        self.dynamic_beam = self.config.get('dynamic_beam', {}).get('enabled', True)
        self.diversity_threshold = self.config.get('diversity_threshold', 0.1)

    def load_config(self) -> None:
        """Load configuration from YAML file with fallbacks."""
        config_name = "strategy_beam.yaml"
        config_path = Path("config/strategies") / config_name

        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Extract parameters from nested structure
            if 'parameters' in yaml_config:
                params = yaml_config['parameters']
                self.config.update(params)

            if 'beam_management' in yaml_config:
                beam_config = yaml_config['beam_management']
                self.config.update(beam_config)

            if 'selection' in yaml_config:
                sel_config = yaml_config['selection']
                self.config.update(sel_config)

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
        """Propose operation using beam search."""
        if self.operations_registry is None:
            return self._fallback_proposal(current_state)

        try:
            # Initialize beam if needed
            if not self.beam or self.beam[0].state.state_id != current_state.state_id:
                self.initialize_beam(current_state)

            # Expand beam
            expanded_beam = self.expand_beam()

            # Prune beam to maintain beam width
            self.prune_beam(expanded_beam)

            # Dynamic beam width adjustment
            if self.dynamic_beam:
                self.adjust_beam_width()

            # Select best operation
            return self.select_best_operation()

        except MemoryError:
            return self._fallback_proposal(current_state)
        except Exception as e:
            return self._fallback_proposal(current_state)

    def _fallback_proposal(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Fallback proposal using greedy selection."""
        if self.operations_registry:
            operations = list(self.operations_registry.operations.keys())
            if operations:
                return (random.choice(operations), {})
        return ('xor_constant', {'constant': random.randint(1, 255)})

    def initialize_beam(self, current_state: State) -> None:
        """Initialize beam with current state."""
        self.beam = []
        initial_beam_state = BeamState(
            state=current_state,
            operations_applied=[],
            score=current_state.score
        )
        self.beam.append(initial_beam_state)

    def expand_beam(self) -> List[BeamState]:
        """Expand beam by applying operations to all beam states."""
        expanded_beam = []

        for beam_state in self.beam:
            # Generate candidate operations
            candidate_operations = self._generate_candidate_operations()

            # Apply each operation
            for operation, params in candidate_operations:
                new_state = self._apply_operation(beam_state.state, operation, params)

                # Create new beam state
                new_operations = beam_state.operations_applied + [(operation, params)]
                new_beam_state = BeamState(
                    state=new_state,
                    operations_applied=new_operations,
                    score=new_state.score
                )

                # Calculate diversity score if needed
                if self.preserve_diversity:
                    new_beam_state.diversity_score = self._calculate_diversity_score(new_beam_state)

                expanded_beam.append(new_beam_state)

        return expanded_beam

    def _generate_candidate_operations(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate candidate operations for expansion."""
        if self.operations_registry:
            operations = list(self.operations_registry.operations.keys())
            random.shuffle(operations)
            selected_ops = operations[:self.branch_factor]
        else:
            default_ops = ['xor_constant', 'rotate_left', 'move_to_front', 'shuffle_bytes']
            selected_ops = random.sample(default_ops, min(self.branch_factor, len(default_ops)))

        # Generate parameters for each operation
        result = []
        for op in selected_ops:
            params = self._generate_operation_params(op)
            result.append((op, params))

        return result

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

    def _apply_operation(self, state: State, operation: str, params: Dict[str, Any]) -> State:
        """Apply operation to create new state (simplified)."""
        # In practice, this would use the operations registry
        # For now, just create a copy with simulated score change
        new_state = state.copy()
        new_state.generation += 1
        new_state.score = state.score + random.uniform(-1, 1)  # Simulated score change
        return new_state

    def prune_beam(self, expanded_beam: List[BeamState]) -> None:
        """Prune beam to keep only top-k states."""
        if not expanded_beam:
            return

        # Normalize scores if enabled
        if self.normalize_scores:
            self._normalize_scores(expanded_beam)

        # Sort by score (higher is better)
        expanded_beam.sort(key=lambda x: x.score, reverse=True)

        # Apply selection method
        if self.selection_method == 'top_n':
            # Simple top-n selection
            selected_beam = expanded_beam[:self.beam_width]
        elif self.selection_method == 'diverse':
            # Diversity-aware selection
            selected_beam = self._diverse_selection(expanded_beam)
        elif self.selection_method == 'aged':
            # Age-aware selection
            selected_beam = self._aged_selection(expanded_beam)
        else:
            # Default to top_n
            selected_beam = expanded_beam[:self.beam_width]

        self.beam = selected_beam

    def _normalize_scores(self, beam_states: List[BeamState]) -> None:
        """Normalize scores for fair comparison."""
        if not beam_states:
            return

        scores = [state.score for state in beam_states]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same, set to 0.5 for all
            for state in beam_states:
                state.score = 0.5
        else:
            # Normalize to [0, 1]
            for state in beam_states:
                state.score = (state.score - min_score) / (max_score - min_score)

    def _diverse_selection(self, beam_states: List[BeamState]) -> List[BeamState]:
        """Select diverse states to avoid local optima."""
        if not beam_states:
            return []

        selected = []
        remaining = beam_states.copy()

        # Always include the best state
        selected.append(remaining[0])
        remaining.pop(0)

        # Select remaining states based on diversity
        while len(selected) < self.beam_width and remaining:
            best_candidate = None
            best_diversity_score = -float('inf')

            for candidate in remaining:
                # Calculate minimum diversity to selected states
                min_diversity = float('inf')
                for selected_state in selected:
                    diversity = self._calculate_state_diversity(candidate, selected_state)
                    min_diversity = min(min_diversity, diversity)

                # Combine score and diversity
                combined_score = candidate.score * 0.7 + min_diversity * 0.3

                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def _aged_selection(self, beam_states: List[BeamState]) -> List[BeamState]:
        """Age-aware selection that prefers newer states."""
        aged_beam = []
        for state in beam_states:
            # Apply age penalty
            age_penalty = state.age * 0.01
            aged_score = state.score - age_penalty

            aged_state = state.copy()
            aged_state.score = aged_score
            aged_beam.append(aged_state)

        aged_beam.sort(key=lambda x: x.score, reverse=True)
        return aged_beam[:self.beam_width]

    def _calculate_diversity_score(self, beam_state: BeamState) -> float:
        """Calculate diversity score for a beam state."""
        if not self.beam:
            return 1.0

        min_diversity = float('inf')
        for existing_state in self.beam:
            diversity = self._calculate_state_diversity(beam_state, existing_state)
            min_diversity = min(min_diversity, diversity)

        return min_diversity

    def _calculate_state_diversity(self, state1: BeamState, state2: BeamState) -> float:
        """Calculate diversity between two beam states."""
        # Simple diversity based on operation sequence difference
        ops1 = [op[0] for op in state1.operations_applied]
        ops2 = [op[0] for op in state2.operations_applied]

        if not ops1 and not ops2:
            return 0.0

        # Calculate Jaccard distance
        set1 = set(ops1)
        set2 = set(ops2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return 1.0 - (intersection / union)

    def adjust_beam_width(self) -> None:
        """Dynamically adjust beam width based on performance."""
        if len(self.beam) < 2:
            return

        # Calculate improvement rate
        current_scores = [state.score for state in self.beam]
        avg_current_score = sum(current_scores) / len(current_scores)

        # Keep track of previous scores (simplified)
        if not hasattr(self, 'previous_avg_score'):
            self.previous_avg_score = avg_current_score
            return

        improvement_rate = (avg_current_score - self.previous_avg_score) / max(1, abs(self.previous_avg_score))
        self.previous_avg_score = avg_current_score

        # Adjust beam width
        if improvement_rate > 0.01:
            # Increasing performance, expand beam
            self.beam_width = min(self.beam_width + 1, self.max_beam_width)
        elif improvement_rate < -0.01:
            # Decreasing performance, shrink beam
            self.beam_width = max(self.beam_width - 1, self.min_beam_width)

    def select_best_operation(self) -> Tuple[str, Dict[str, Any]]:
        """Select best operation from beam."""
        if not self.beam:
            return ('xor_constant', {'constant': 1})

        # Get the best beam state
        best_beam_state = max(self.beam, key=lambda x: x.score)

        # Return the last operation that led to this state
        last_operation = best_beam_state.get_last_operation()
        if last_operation:
            return last_operation

        # Fallback
        return ('xor_constant', {'constant': 1})

    def accept(self, new_state: State) -> bool:
        """Accept state based on beam criteria."""
        # Age all beam states
        for beam_state in self.beam:
            beam_state.age += 1

        # Standard acceptance criteria
        return new_state.score > self.best_score