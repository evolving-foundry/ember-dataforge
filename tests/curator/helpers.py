"""
Test helper module - provides ray-related helper functions
"""
import pytest
import sys
import os

# Try to import ray; return None if it fails
try:
    import ray
    HAS_RAY = True
except ImportError:
    ray = None
    HAS_RAY = False


def skip_if_no_ray(reason="Ray is not installed"):
    """Decorator: skip tests if ray is unavailable"""
    def decorator(func):
        if not HAS_RAY:
            return pytest.mark.skip(reason=reason)(func)
        return func
    return decorator


def get_ray():
    """Get the ray module"""
    return ray


def is_ray_available():
    """Check whether ray is available"""
    return HAS_RAY
