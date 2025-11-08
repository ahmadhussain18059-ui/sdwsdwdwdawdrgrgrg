"""
Base strategy interface for BSEE search strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from bsee.engine.state import State
from bsee.utils.error_handler import safe_execute, get_global_error_handler, ErrorSeverity


class BaseStrategy(ABC):
    """Abstract base class for search strategies."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration."""
        self.config = config
        self.name = self.__class__.__name__
        self.iteration_count = 0
        self.best_score = float('-inf')
        self.no_improvement_count = 0
        self.converged = False

        # Error handling
        self.error_handler = get_global_error_handler()
        self.error_count = 0
        self.recovery_count = 0
        self.last_error = None

    @abstractmethod
    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose next operation to apply.

        Args:
            current_state: Current state of the binary data

        Returns:
            Tuple of (operation_name, operation_parameters)
        """
        pass

    @abstractmethod
    def accept(self, new_state: State) -> bool:
        """Decide whether to accept a new state.

        Args:
            new_state: Proposed new state

        Returns:
            True if the state should be accepted, False otherwise
        """
        pass

    def is_converged(self) -> bool:
        """Check if the search has converged."""
        return self.converged

    def reset(self) -> None:
        """Reset the strategy state."""
        self.iteration_count = 0
        self.best_score = float('-inf')
        self.no_improvement_count = 0
        self.converged = False

    def update_statistics(self, state: State, accepted: bool) -> None:
        """Update strategy statistics based on state evaluation."""
        self.iteration_count += 1

        if accepted and state.score > self.best_score:
            self.best_score = state.score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Check convergence criteria
        self._check_convergence()

    def _check_convergence(self) -> None:
        """Check if convergence criteria are met."""
        max_no_improvement = self.config.get('max_no_improvement', 50)
        max_iterations = self.config.get('max_iterations', 1000)

        if self.no_improvement_count >= max_no_improvement:
            self.converged = True

        if self.iteration_count >= max_iterations:
            self.converged = True

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy's current state."""
        return {
            'name': self.name,
            'iteration_count': self.iteration_count,
            'best_score': self.best_score,
            'no_improvement_count': self.no_improvement_count,
            'converged': self.converged,
            'error_count': self.error_count,
            'recovery_count': self.recovery_count,
            'last_error': str(self.last_error) if self.last_error else None
        }

    def safe_propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """
        Safely propose an operation with error handling and recovery.

        Returns:
            Tuple of (operation_name, operation_parameters)
            If proposal fails, returns a safe fallback operation
        """
        try:
            return self.propose(current_state)
        except Exception as e:
            self.error_count += 1
            self.last_error = e

            # Handle the error
            error_context = {
                'strategy': self.name,
                'iteration': self.iteration_count,
                'current_score': current_state.score
            }

            error_result = self.error_handler.handle_error(e, error_context, ErrorSeverity.MEDIUM)

            if error_result['recovery_successful']:
                self.recovery_count += 1
                # Try again after recovery
                try:
                    return self.propose(current_state)
                except Exception:
                    pass  # Fall through to fallback

            # Fallback strategy
            return self._get_fallback_proposal(current_state)

    def safe_accept(self, new_state: State) -> bool:
        """
        Safely decide whether to accept a new state with error handling.

        Returns:
            True if the state should be accepted, False otherwise
            Default to False on errors to be safe
        """
        try:
            return self.accept(new_state)
        except Exception as e:
            self.error_count += 1
            self.last_error = e

            error_context = {
                'strategy': self.name,
                'iteration': self.iteration_count,
                'proposed_score': new_state.score
            }

            self.error_handler.handle_error(e, error_context, ErrorSeverity.MEDIUM)

            # Default to conservative behavior on errors
            return False

    def _get_fallback_proposal(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """
        Get a fallback proposal when the main strategy fails.

        Can be overridden by subclasses for strategy-specific fallbacks
        """
        # Simple fallback: apply a basic XOR operation
        return ('xor_constant', {'constant': 1})

    def handle_strategy_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Handle strategy-specific errors.

        Args:
            error: The exception that occurred
            context: Additional context about the error

        Returns:
            True if recovery was successful, False otherwise
        """
        self.error_count += 1
        self.last_error = error

        error_context = context or {}
        error_context.update({
            'strategy': self.name,
            'iteration': self.iteration_count
        })

        error_result = self.error_handler.handle_error(error, error_context)

        if error_result['recovery_successful']:
            self.recovery_count += 1

        return error_result['recovery_successful']