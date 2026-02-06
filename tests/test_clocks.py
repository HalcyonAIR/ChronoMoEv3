"""
Tests for three-timescale clock system.
"""

import pytest
import math
from chronomoe_v3.clocks import (
    ClockConfig,
    ThreeClockEMA,
    compute_alpha_from_half_life,
    compute_half_life_from_alpha,
)


class TestClockConfig:
    """Test ClockConfig initialization and properties."""

    def test_init_from_half_life(self):
        """Initialize from half_life, compute alpha."""
        clock = ClockConfig(name="test", half_life=10)

        assert clock.half_life == 10
        assert clock.alpha is not None
        assert 0 < clock.alpha < 1

        # After half_life steps, signal should decay to ~50%
        value = 1.0
        for _ in range(clock.half_life):
            value = clock.ema_update(value, 0.0)

        assert 0.48 < value < 0.52  # Approximately 0.5

    def test_init_from_alpha(self):
        """Initialize from alpha, compute half_life."""
        clock = ClockConfig(name="test", alpha=0.9)

        assert clock.alpha == 0.9
        assert clock.half_life is not None
        assert clock.half_life > 0

    def test_alpha_half_life_consistency(self):
        """Alpha and half_life should be consistent."""
        clock1 = ClockConfig(name="test1", half_life=100)
        clock2 = ClockConfig(name="test2", alpha=clock1.alpha)

        assert abs(clock1.half_life - clock2.half_life) <= 1  # Within rounding

    def test_ema_update(self):
        """Test EMA update formula."""
        clock = ClockConfig(name="test", alpha=0.9)

        old = 0.5
        new = 1.0
        updated = clock.ema_update(old, new)

        expected = 0.9 * 0.5 + 0.1 * 1.0
        assert abs(updated - expected) < 1e-6

    def test_decay_properties(self):
        """Test decay rate property."""
        clock = ClockConfig(name="test", alpha=0.9)

        assert clock.retention == 0.9
        assert clock.decay_rate == 0.1

    def test_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError):
            ClockConfig(name="test", alpha=0.0)

        with pytest.raises(ValueError):
            ClockConfig(name="test", alpha=1.0)

        with pytest.raises(ValueError):
            ClockConfig(name="test", alpha=1.5)


class TestThreeClockEMA:
    """Test ThreeClockEMA system."""

    def test_initialization(self):
        """Test default initialization."""
        clocks = ThreeClockEMA()

        assert clocks.fast == 0.0
        assert clocks.mid == 0.0
        assert clocks.slow == 0.0

    def test_single_update(self):
        """Test single update to all clocks."""
        clocks = ThreeClockEMA(alpha_fast=0.9, alpha_mid=0.99, alpha_slow=0.999)

        clocks.update(1.0)

        # All should move toward 1.0, fast moving most
        assert clocks.fast > clocks.mid > clocks.slow > 0
        assert clocks.fast == pytest.approx(0.1, abs=1e-6)
        assert clocks.mid == pytest.approx(0.01, abs=1e-6)
        assert clocks.slow == pytest.approx(0.001, abs=1e-6)

    def test_convergence(self):
        """Test convergence to steady signal."""
        clocks = ThreeClockEMA(alpha_fast=0.9, alpha_mid=0.99, alpha_slow=0.999)

        # Feed constant signal
        for _ in range(10000):
            clocks.update(0.8)

        # All should converge to 0.8
        assert abs(clocks.fast - 0.8) < 0.01
        assert abs(clocks.mid - 0.8) < 0.01
        assert abs(clocks.slow - 0.8) < 0.01

    def test_fast_response_to_change(self):
        """Test fast clock responds quickly to change."""
        clocks = ThreeClockEMA(alpha_fast=0.9, alpha_mid=0.99, alpha_slow=0.999)

        # Build up to 1.0
        for _ in range(1000):
            clocks.update(1.0)

        # Sudden drop
        for _ in range(10):
            clocks.update(0.0)

        # Fast should drop significantly, slow should barely move
        assert clocks.fast < 0.5
        assert clocks.mid > 0.8
        assert clocks.slow > 0.98

    def test_delta_properties(self):
        """Test delta computation."""
        clocks = ThreeClockEMA(alpha_fast=0.9, alpha_mid=0.99, alpha_slow=0.999)

        # Build up
        for _ in range(1000):
            clocks.update(1.0)

        # Sudden drop
        for _ in range(10):
            clocks.update(0.0)

        # Deltas should be negative (degradation)
        assert clocks.delta_fast_slow < 0
        assert clocks.delta_mid_slow < 0

    def test_reset(self):
        """Test reset functionality."""
        clocks = ThreeClockEMA()

        clocks.update(1.0)
        assert clocks.fast > 0

        clocks.reset(0.5)
        assert clocks.fast == 0.5
        assert clocks.mid == 0.5
        assert clocks.slow == 0.5


class TestHelperFunctions:
    """Test helper functions."""

    def test_alpha_half_life_roundtrip(self):
        """Test conversion roundtrip."""
        half_life = 100
        alpha = compute_alpha_from_half_life(half_life)
        half_life_recovered = compute_half_life_from_alpha(alpha)

        assert abs(half_life - half_life_recovered) <= 1

    def test_alpha_values(self):
        """Test expected alpha values for common half-lives."""
        # half_life=10 → alpha≈0.933
        alpha = compute_alpha_from_half_life(10)
        assert 0.93 < alpha < 0.94

        # half_life=100 → alpha≈0.993
        alpha = compute_alpha_from_half_life(100)
        assert 0.992 < alpha < 0.994

        # half_life=1000 → alpha≈0.9993
        alpha = compute_alpha_from_half_life(1000)
        assert 0.9992 < alpha < 0.9994


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
