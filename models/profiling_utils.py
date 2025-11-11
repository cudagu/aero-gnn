"""
Profiling utilities for MeshGraphNet layers.
Tracks timing and memory usage for different operations without breaking existing code.
"""

import torch
import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Optional, Dict
import numpy as np


class LayerProfiler:
    """
    Profiler for tracking timing and memory usage in neural network layers.
    Can be enabled/disabled globally without changing code structure.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.stats = defaultdict(lambda: {'times': [], 'memory': [], 'count': 0})
        self.device = None

    def reset(self):
        """Reset all collected statistics."""
        self.stats.clear()

    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling a code block.

        Usage:
            with profiler.profile("edge_gather"):
                # code to profile
                pass
        """
        if not self.enabled:
            yield
            return

        # Synchronize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Record start time and memory
        start_time = time.perf_counter()
        start_mem = self._get_memory()

        try:
            yield
        finally:
            # Synchronize CUDA and record end time
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            end_mem = self._get_memory()

            # Store statistics
            elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
            mem_delta = end_mem - start_mem

            self.stats[name]['times'].append(elapsed)
            self.stats[name]['memory'].append(mem_delta)
            self.stats[name]['count'] += 1

    def _get_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0

    def get_summary(self) -> Dict:
        """
        Get summary statistics for all profiled operations.

        Returns:
            Dictionary with mean, std, min, max for timing and memory
        """
        summary = {}

        for name, data in self.stats.items():
            times = np.array(data['times'])
            memory = np.array(data['memory'])

            summary[name] = {
                'count': data['count'],
                'time_mean_ms': np.mean(times) if len(times) > 0 else 0,
                'time_std_ms': np.std(times) if len(times) > 0 else 0,
                'time_min_ms': np.min(times) if len(times) > 0 else 0,
                'time_max_ms': np.max(times) if len(times) > 0 else 0,
                'time_total_ms': np.sum(times) if len(times) > 0 else 0,
                'memory_mean_mb': np.mean(memory) if len(memory) > 0 else 0,
                'memory_std_mb': np.std(memory) if len(memory) > 0 else 0,
                'memory_min_mb': np.min(memory) if len(memory) > 0 else 0,
                'memory_max_mb': np.max(memory) if len(memory) > 0 else 0,
            }

        return summary

    def print_summary(self, sort_by: str = 'time_total_ms'):
        """
        Print a formatted summary of profiling statistics.

        Args:
            sort_by: Key to sort operations by ('time_total_ms', 'time_mean_ms', 'memory_mean_mb', etc.)
        """
        if not self.stats:
            print("No profiling data collected.")
            return

        summary = self.get_summary()

        # Sort operations
        sorted_ops = sorted(summary.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)

        print("\n" + "="*100)
        print(f"{'Operation':<30} {'Count':>8} {'Time (ms)':>20} {'Memory (MB)':>20}")
        print(f"{'':30} {'':8} {'Mean±Std (Total)':>20} {'Mean±Std':>20}")
        print("="*100)

        for name, stats in sorted_ops:
            time_str = f"{stats['time_mean_ms']:.3f}±{stats['time_std_ms']:.3f} ({stats['time_total_ms']:.2f})"
            mem_str = f"{stats['memory_mean_mb']:.2f}±{stats['memory_std_mb']:.2f}"
            print(f"{name:<30} {stats['count']:>8} {time_str:>20} {mem_str:>20}")

        print("="*100)

        # Calculate total time
        total_time = sum(s['time_total_ms'] for s in summary.values())
        print(f"\nTotal profiled time: {total_time:.2f} ms")

        # Show percentage breakdown
        print(f"\n{'Time breakdown:':<30}")
        for name, stats in sorted_ops:
            percentage = (stats['time_total_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"  {name:<28} {percentage:>5.1f}%")
        print()


# Global profiler instance - can be enabled/disabled globally
_global_profiler = LayerProfiler(enabled=False)


def enable_profiling():
    """Enable profiling globally."""
    _global_profiler.enabled = True
    _global_profiler.reset()


def disable_profiling():
    """Disable profiling globally."""
    _global_profiler.enabled = False


def reset_profiling():
    """Reset profiling statistics."""
    _global_profiler.reset()


def get_profiler() -> LayerProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def print_profiling_summary(sort_by: str = 'time_total_ms'):
    """Print profiling summary."""
    _global_profiler.print_summary(sort_by=sort_by)


def get_profiling_summary() -> Dict:
    """Get profiling summary as dictionary."""
    return _global_profiler.get_summary()
