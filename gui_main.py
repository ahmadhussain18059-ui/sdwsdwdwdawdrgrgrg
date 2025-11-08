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
    """Check if required dependencies are available."""
    required_modules = [
        'tkinter',
        'matplotlib',
        'numpy',
        'scipy',
        'yaml'
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print("Error: Missing required dependencies:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)


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
    """Main GUI entry point."""
    print("BSEE - Binary Structure Exploration Engine")
    print("GUI Version 1.0.0")
    print("=" * 50)

    # Check requirements
    check_python_version()
    check_dependencies()
    setup_directories()

    try:
        # Create and run GUI
        app = MainWindow()
        app.run()

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()