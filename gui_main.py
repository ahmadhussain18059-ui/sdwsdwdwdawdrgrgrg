#!/usr/bin/env python3
"""
BSEE GUI Entry Point

Binary Structure Exploration Engine - Windows GUI Application
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Graceful GUI dependency handling
GUI_AVAILABLE = True
GUI_FALLBACK_MESSAGE = ""

try:
    import tkinter as tk
    from gui.main_window import MainWindow
except ImportError as e:
    GUI_AVAILABLE = False
    GUI_FALLBACK_MESSAGE = str(e)
    print(f"Warning: GUI modules not available: {e}")
    print("Falling back to CLI mode...")

    # Check if we can provide CLI fallback
    try:
        from main import main as cli_main
        CLI_FALLBACK_AVAILABLE = True
    except ImportError:
        CLI_FALLBACK_AVAILABLE = False
        print("Error: CLI fallback also not available")
        print("Please install required dependencies:")
        print("pip install -r requirements.txt")


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: BSEE GUI requires Python 3.8 or higher")
        print(f"Current version: {sys.version}")
        sys.exit(1)


def check_dependencies():
    """Check if required dependencies are available with graceful fallback."""
    # Core dependencies required for any mode
    core_modules = ['yaml', 'numpy']

    # GUI-specific dependencies
    gui_modules = ['tkinter', 'matplotlib']

    # Optional dependencies for enhanced functionality
    optional_modules = ['scipy']

    missing_core = []
    missing_gui = []
    missing_optional = []

    # Check core dependencies
    for module in core_modules:
        try:
            __import__(module)
        except ImportError:
            missing_core.append(module)

    # Check GUI dependencies
    for module in gui_modules:
        try:
            __import__(module)
        except ImportError:
            missing_gui.append(module)

    # Check optional dependencies
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)

    # Handle missing core dependencies (critical)
    if missing_core:
        print("Error: Missing critical dependencies:")
        for module in missing_core:
            print(f"  - {module}")
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")
        return False

    # Handle missing GUI dependencies
    if missing_gui:
        global GUI_AVAILABLE, GUI_FALLBACK_MESSAGE
        GUI_AVAILABLE = False
        GUI_FALLBACK_MESSAGE = f"Missing GUI modules: {', '.join(missing_gui)}"
        print(f"Warning: {GUI_FALLBACK_MESSAGE}")
        print("Falling back to CLI mode...")

        # Check if CLI fallback is available
        try:
            from main import main as cli_main
            CLI_FALLBACK_AVAILABLE = True
        except ImportError:
            CLI_FALLBACK_AVAILABLE = False
            print("Error: CLI fallback also not available")
            return False

    # Handle missing optional dependencies
    if missing_optional:
        print(f"Warning: Missing optional modules: {', '.join(missing_optional)}")
        print("Some advanced features may not be available")

    return True


def setup_directories():
    """Create necessary directories."""
    directories = [
        'inputs',
        'results',
        'presets',
        'history',
        'logs',
        'temp'
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def main():
    """Main entry point with GUI/CLI fallback support."""
    print("BSEE - Binary Structure Exploration Engine")
    print("Version 1.0.0")
    print("=" * 50)

    # Check requirements
    check_python_version()
    if not check_dependencies():
        sys.exit(1)

    setup_directories()

    try:
        if GUI_AVAILABLE:
            print("Starting GUI mode...")
            # Create and run GUI
            app = MainWindow()
            app.run()
        else:
            print("Starting CLI fallback mode...")
            print("Note: GUI mode not available due to missing dependencies")
            print(f"Reason: {GUI_FALLBACK_MESSAGE}")
            print()

            # Import and run CLI main with simulated arguments
            from main import main as cli_main, parse_arguments
            import argparse

            # Create a simple CLI interface when GUI is not available
            try:
                # Try to get arguments from command line, or provide simple defaults
                if len(sys.argv) > 1:
                    # Forward arguments to CLI main
                    cli_main()
                else:
                    # Provide simple interactive CLI interface
                    run_interactive_cli()
            except SystemExit:
                # Handle sys.exit() from CLI main gracefully
                pass

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_interactive_cli():
    """Run a simple interactive CLI interface."""
    print("Interactive CLI Mode - BSEE")
    print("-" * 30)
    print()

    # Get input file
    while True:
        input_file = input("Enter path to binary file: ").strip()
        if input_file and Path(input_file).exists():
            break
        print(f"File not found: {input_file}")

    # Get strategy
    strategies = ['greedy', 'beam', 'annealing', 'mcts', 'genetic', 'heuristic']
    print(f"Available strategies: {', '.join(strategies)}")
    while True:
        strategy = input("Enter strategy (default: greedy): ").strip() or 'greedy'
        if strategy in strategies:
            break
        print(f"Invalid strategy. Choose from: {', '.join(strategies)}")

    # Create simulated CLI arguments
    class SimulatedArgs:
        def __init__(self):
            self.input_file = input_file
            self.policy = "config/policies/policy_ideality.yaml"
            self.costs = "config/costs/cost_default.yaml"
            self.strategy = strategy
            self.metrics = "file_ideality_score,entropy_global,lz77_ratio"
            self.target_metrics = "file_ideality_score=max,entropy_global=min"
            self.max_operations = 100
            self.max_cost = 1000
            self.allowed_ops = None
            self.operation_limit = None
            self.output_dir = "results"
            self.log_level = "INFO"

    # Update sys.argv for CLI main
    original_argv = sys.argv.copy()
    sys.argv = [
        'bsee',
        input_file,
        '--strategy', strategy,
        '--max-operations', '100',
        '--max-cost', '1000'
    ]

    try:
        # Run CLI main
        from main import main as cli_main
        cli_main()
    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()