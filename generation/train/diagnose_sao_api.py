#!/usr/bin/env python3
"""
Diagnostic script to find the correct stable-audio-tools API for model loading.

Run this on the server:
    source ~/stableaudio/venv/bin/activate
    python diagnose_sao_api.py
"""

import sys
import inspect

print("="*70)
print("Stable Audio Tools API Diagnostic")
print("="*70 + "\n")

# Import stable_audio_tools
try:
    import stable_audio_tools as sat
    print(f"✓ stable-audio-tools imported successfully")
    print(f"  Version: {sat.__version__ if hasattr(sat, '__version__') else 'unknown'}")
    print(f"  Location: {sat.__file__}\n")
except ImportError as e:
    print(f"✗ Failed to import stable-audio-tools: {e}\n")
    sys.exit(1)

# Explore module structure
print("Available top-level attributes in stable_audio_tools:")
attrs = [attr for attr in dir(sat) if not attr.startswith('_')]
for attr in sorted(attrs)[:30]:  # First 30
    try:
        obj = getattr(sat, attr)
        obj_type = type(obj).__name__
        print(f"  - {attr} ({obj_type})")
    except Exception as e:
        print(f"  - {attr} (ERROR: {e})")

print("\n" + "="*70)
print("Looking for model loading functions...")
print("="*70 + "\n")

# Try to find model-related functions
search_terms = ['model', 'load', 'get', 'pretrained', 'checkpoint']
found = False

for attr_name in dir(sat):
    if any(term in attr_name.lower() for term in search_terms):
        try:
            attr = getattr(sat, attr_name)
            if callable(attr):
                print(f"Found callable: {attr_name}")
                print(f"  Signature: {inspect.signature(attr)}")
                print(f"  Docstring: {inspect.getdoc(attr)[:200] if inspect.getdoc(attr) else 'No docstring'}\n")
                found = True
        except Exception as e:
            pass

if not found:
    print("No model loading functions found at top level.\n")

# Try to import and explore known submodules
print("="*70)
print("Exploring known submodules...")
print("="*70 + "\n")

submodules_to_try = [
    'stable_audio_tools.models',
    'stable_audio_tools.model',
    'stable_audio_tools.inference',
    'stable_audio_tools.training',
    'stable_audio_tools.utils',
]

for submodule_name in submodules_to_try:
    try:
        submodule = __import__(submodule_name, fromlist=[''])
        print(f"✓ {submodule_name} exists")
        print(f"  Attributes: {[a for a in dir(submodule) if not a.startswith('_')][:10]}\n")
    except ImportError:
        print(f"✗ {submodule_name} does not exist\n")

print("="*70)
print("Recommendations")
print("="*70 + "\n")
print("1. Check if stable-audio-tools provides a training CLI:")
print("   python -m stable_audio_tools --help\n")
print("2. Look for training examples in:")
print("   https://github.com/Stability-AI/stable-audio-tools\n")
print("3. Check the GitHub wiki tutorial you're using for exact code\n")
