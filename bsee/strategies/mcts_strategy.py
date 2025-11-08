"""
Monte Carlo Tree Search strategy for BSEE.
"""

import random
import math
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State


class TreeNode:
    """Node in the MCTS search tree."""

    def __init__(self, state: State, parent: Optional['TreeNode'] = None):
        self.state = state  # State object
        self.parent = parent
        self.children = {}  # operation_name -> TreeNode
        self.visits = 0
        self.value = 0.0
        self.untried_operations = []  # List of operations not yet tried

    def is_fully_expanded(self) -> bool:
        """Check if all possible operations have been tried."""
        return len(self.untried_operations) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.state.generation >= 50  # Max depth limit

    def get_best_child(self, exploration_constant: float = 1.4) -> 'TreeNode':
        """Get the best child using UCT formula."""
        best_child = None
        best_uct = -float('inf')

        for child in self.children.values():
            if child.visits == 0:
                uct = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
                uct = exploitation + exploration

            if uct > best_uct:
                best_uct = uct
                best_child = child

        return best_child

    def update(self, value: float) -> None:
        """Update node statistics."""
        self.visits += 1
        self.value += value


class MCTSStrategy(BaseStrategy):
    """Monte Carlo Tree Search strategy."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize MCTS strategy."""
        super().__init__(config)
        self.load_config()
        self.root_node = None
        self.operations_registry = None  # Will be set by the pipeline

        # Configuration parameters
        self.exploration_constant = self.config.get('exploration_constant', 1.4)
        self.simulation_count = self.config.get('simulation_count', 100)
        self.rollout_depth = self.config.get('rollout_depth', 10)
        self.max_tree_depth = self.config.get('max_tree_depth', 50)
        self.max_tree_nodes = self.config.get('max_tree_nodes', 10000)
        self.prune_tree = self.config.get('prune_tree', True)
        self.prune_threshold = self.config.get('prune_threshold', 0.01)
        self.final_selection = self.config.get('final_selection', 'most_visited')

    def load_config(self) -> None:
        """Load configuration from YAML file with fallbacks."""
        config_name = "strategy_mcts.yaml"
        config_path = Path("config/strategies") / config_name

        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Extract parameters from nested structure
            if 'parameters' in yaml_config:
                params = yaml_config['parameters']
                self.config.update(params)

            if 'tree_management' in yaml_config:
                tree_config = yaml_config['tree_management']
                self.config.update(tree_config)

            if 'simulation' in yaml_config:
                sim_config = yaml_config['simulation']
                self.config.update(sim_config)

            if 'selection' in yaml_config:
                sel_config = yaml_config['selection']
                self.config.update(sel_config)

            if 'memory' in yaml_config:
                mem_config = yaml_config['memory']
                self.config.update(mem_config)

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
        """Propose operation using MCTS."""
        if self.operations_registry is None:
            # Fallback to random choice if no operations registry
            return self._fallback_proposal(current_state)

        try:
            # Initialize or update root node
            if self.root_node is None or self.root_node.state.state_id != current_state.state_id:
                self.root_node = TreeNode(current_state)
                self._initialize_untried_operations(self.root_node)

            # Run MCTS simulations
            for _ in range(self.simulation_count):
                self._mcts_iteration()

            # Memory management - prune tree if too large
            if self.prune_tree and self._count_tree_nodes() > self.max_tree_nodes:
                self._prune_tree()

            # Get best move
            best_operation = self.get_best_move()

            return best_operation

        except MemoryError:
            # Handle memory errors gracefully
            return self._fallback_proposal(current_state)
        except Exception as e:
            # Handle other errors
            return self._fallback_proposal(current_state)

    def _fallback_proposal(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Fallback proposal using greedy selection."""
        if self.operations_registry:
            operations = list(self.operations_registry.operations.keys())
            if operations:
                return (random.choice(operations), {})
        return ('xor_constant', {'constant': random.randint(1, 255)})

    def _initialize_untried_operations(self, node: TreeNode) -> None:
        """Initialize list of untried operations for a node."""
        if self.operations_registry:
            node.untried_operations = list(self.operations_registry.operations.keys())
        else:
            node.untried_operations = ['xor_constant', 'rotate_left', 'move_to_front', 'shuffle_bytes']

    def _mcts_iteration(self) -> None:
        """Perform one MCTS iteration: selection, expansion, simulation, backpropagation."""
        # Selection
        node = self._select_node(self.root_node)

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = self._expand_node(node)

        # Simulation
        reward = self._simulate(node)

        # Backpropagation
        self._backpropagate(node, reward)

    def _select_node(self, node: TreeNode) -> TreeNode:
        """Select node for expansion using tree traversal."""
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.get_best_child(self.exploration_constant)
        return node

    def _expand_node(self, node: TreeNode) -> TreeNode:
        """Expand node by trying an untried operation."""
        if not node.untried_operations:
            return node

        operation = random.choice(node.untried_operations)
        node.untried_operations.remove(operation)

        # Generate parameters for the operation
        params = self._generate_operation_params(operation)

        # Apply operation to get new state (simplified - in practice, this would use the operations registry)
        new_state = self._apply_operation(node.state, operation, params)

        # Create child node
        child_node = TreeNode(new_state, parent=node)
        child_node.untried_operations = node.untried_operations.copy()

        node.children[operation] = child_node
        return child_node

    def _simulate(self, node: TreeNode) -> float:
        """Run random simulation from node."""
        current_state = node.state
        total_reward = 0.0

        for _ in range(self.rollout_depth):
            if current_state.generation >= self.max_tree_depth:
                break

            # Random operation selection for simulation
            if self.operations_registry:
                operation = random.choice(list(self.operations_registry.operations.keys()))
            else:
                operation = random.choice(['xor_constant', 'rotate_left', 'move_to_front', 'shuffle_bytes'])

            params = self._generate_operation_params(operation)
            current_state = self._apply_operation(current_state, operation, params)

            # Accumulate reward based on score improvement
            total_reward += current_state.score

        return total_reward / max(1, self.rollout_depth)

    def _backpropagate(self, node: TreeNode, reward: float) -> None:
        """Backpropagate reward up the tree."""
        while node is not None:
            node.update(reward)
            node = node.parent

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
        # For now, just create a copy with incremented generation
        new_state = state.copy()
        new_state.generation += 1
        new_state.score = state.score + random.uniform(-1, 1)  # Simulated score change
        return new_state

    def _count_tree_nodes(self) -> int:
        """Count total nodes in the tree."""
        if self.root_node is None:
            return 0
        return self._count_subtree_nodes(self.root_node)

    def _count_subtree_nodes(self, node: TreeNode) -> int:
        """Count nodes in a subtree."""
        count = 1
        for child in node.children.values():
            count += self._count_subtree_nodes(child)
        return count

    def _prune_tree(self) -> None:
        """Prune tree to reduce memory usage."""
        if self.root_node is None:
            return

        # Simple pruning: remove low-value children
        self._prune_subtree(self.root_node)

    def _prune_subtree(self, node: TreeNode) -> None:
        """Prune a subtree."""
        children_to_remove = []
        for op_name, child in node.children.items():
            if child.visits > 0 and child.value / child.visits < self.prune_threshold:
                children_to_remove.append(op_name)
            else:
                self._prune_subtree(child)

        for op_name in children_to_remove:
            del node.children[op_name]

    def get_best_move(self) -> Tuple[str, Dict[str, Any]]:
        """Get the best operation from the root node."""
        if self.root_node is None or not self.root_node.children:
            return self._fallback_proposal(self.root_node.state if self.root_node else None)

        best_child = None
        best_operation = None
        best_params = {}

        if self.final_selection == 'most_visited':
            # Select most visited child
            max_visits = -1
            for op_name, child in self.root_node.children.items():
                if child.visits > max_visits:
                    max_visits = child.visits
                    best_child = child
                    best_operation = op_name
        elif self.final_selection == 'highest_value':
            # Select child with highest average value
            best_avg_value = -float('inf')
            for op_name, child in self.root_node.children.items():
                if child.visits > 0:
                    avg_value = child.value / child.visits
                    if avg_value > best_avg_value:
                        best_avg_value = avg_value
                        best_child = child
                        best_operation = op_name
        else:  # robust selection
            # Select child with both good visit count and good value
            best_score = -float('inf')
            for op_name, child in self.root_node.children.items():
                if child.visits > 10:  # Minimum visits threshold
                    score = (child.value / child.visits) * math.log(child.visits)
                    if score > best_score:
                        best_score = score
                        best_child = child
                        best_operation = op_name

        if best_operation:
            best_params = self._generate_operation_params(best_operation)
            return (best_operation, best_params)
        else:
            return self._fallback_proposal(self.root_node.state)

    def accept(self, new_state: State) -> bool:
        """Accept based on MCTS evaluation."""
        # Update root node if this state was reached through MCTS
        if self.root_node and new_state.state_id in [child.state.state_id for child in self.root_node.children.values()]:
            for op_name, child in self.root_node.children.items():
                if child.state.state_id == new_state.state_id:
                    self.root_node = child
                    break

        return new_state.score > self.best_score