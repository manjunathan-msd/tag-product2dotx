import os
import importlib

# Get the directory of the current file (i.e., mymodule/)
module_dir = os.path.dirname(__file__)

# Iterate over all files in the module directory
for filename in os.listdir(module_dir):
    # Check if the file is a Python file and not __init__.py
    if filename.endswith('.py') and filename != '__init__.py':
        # Get the module name by stripping the .py extension
        module_name = filename[:-3]
        # Import the module dynamically
        module = importlib.import_module(f'.{module_name}', package=__name__)
        # Import all attributes (e.g., classes) from the module into the current namespace
        for attribute_name in dir(module):
            if not attribute_name.startswith('_'):
                globals()[attribute_name] = getattr(module, attribute_name)
