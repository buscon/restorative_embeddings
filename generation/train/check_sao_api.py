#!/usr/bin/env python3
"""
Diagnostic script to check stable-audio-tools API structure
"""

print("Checking stable-audio-tools installation...\n")

try:
    import stable_audio_tools
    print(f"✓ stable-audio-tools imported successfully")
    print(f"  Version: {stable_audio_tools.__version__ if hasattr(stable_audio_tools, '__version__') else 'unknown'}")
    print(f"  Location: {stable_audio_tools.__file__}\n")
except ImportError as e:
    print(f"✗ Failed to import stable-audio-tools: {e}\n")
    exit(1)

# Check available modules and functions
print("Available modules in stable-audio-tools:")
import pkgutil
import stable_audio_tools
for importer, modname, ispkg in pkgutil.iter_modules(stable_audio_tools.__path__, stable_audio_tools.__name__ + "."):
    print(f"  - {modname} (package: {ispkg})")

print("\nChecking for common model loading functions:")

# Try different import paths
import_paths = [
    "stable_audio_tools.model.get_pretrained",
    "stable_audio_tools.model.load_model",
    "stable_audio_tools.get_model",
    "stable_audio_tools.inference.get_pretrained",
    "stable_audio_tools.models.get_pretrained",
]

for path in import_paths:
    try:
        parts = path.split('.')
        module_path = '.'.join(parts[:-1])
        func_name = parts[-1]

        module = __import__(module_path, fromlist=[func_name])
        if hasattr(module, func_name):
            print(f"  ✓ Found: {path}")
        else:
            print(f"  ✗ Module exists but no function: {path}")
    except ImportError:
        print(f"  ✗ Not found: {path}")

# List all attributes of stable_audio_tools
print("\nAll attributes of stable_audio_tools:")
attrs = dir(stable_audio_tools)
for attr in attrs:
    if not attr.startswith('_'):
        print(f"  - {attr}")

print("\nTip: Check the stable-audio-tools GitHub repo or documentation for the correct API:")
print("     https://github.com/Stability-AI/stable-audio-tools")
