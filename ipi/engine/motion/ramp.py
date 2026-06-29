"""Holds the algorithms to perform replica exchange.

Algorithms implemented by Robert Meissner and Riccardo Petraglia, 2016
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2016 i-PI developers
# See the "licenses" directory for full license information.

from ipi.engine.motion import Motion

__all__ = ["TemperatureRamp", "PressureRamp"]


def _ramp_value(start, end, ramp_step, total_steps, logscale):
    """Returns the ramped value for a given step within the ramp.

    This keeps the original i-PI ramp convention:
    - ramp_step = 1 gives the first interpolated value
    - ramp_step >= total_steps gives end
    """
    if total_steps <= 0:
        return end

    if ramp_step >= total_steps:
        return end

    frac = ramp_step / float(total_steps)

    if logscale:
        # Logarithmic interpolation requires strictly positive endpoints.
        # If not, fall back to linear interpolation.
        if start > 0.0 and end > 0.0:
            return start * (end / start) ** frac

    return start + (end - start) * frac


def _apply_ramp(motion, attr_name, start, end):
    """Applies a ramp, optionally delayed by wait_steps.

    Backward compatibility rules:
    - wait_steps = 0 reproduces the original implementation exactly
    - during the wait period, the value is held at start
    - after the wait, the ramp continues as if it had simply started later
    """
    motion.current_step += 1

    # Hold the initial value during the waiting period.
    if motion.current_step <= motion.wait_steps:
        setattr(motion.ensemble, attr_name, start)
        return

    # Shift the original ramp by wait_steps calls.
    # For wait_steps=0, this gives ramp_step=1 on the first call,
    # exactly matching the old code.
    ramp_step = motion.current_step - motion.wait_steps

    value = _ramp_value(start, end, ramp_step, motion.total_steps, motion.logscale)
    setattr(motion.ensemble, attr_name, value)


class TemperatureRamp(Motion):
    """Temperature ramp (quench/heat)."""

    def __init__(
        self,
        fixcom=False,
        fixatoms_dof=None,
        t_start=1.0,
        t_end=1.0,
        total_steps=0,
        current_step=0,
        wait_steps=0,
        logscale=True,
    ):
        super(TemperatureRamp, self).__init__()
        self.t_start = t_start
        self.t_end = t_end
        self.total_steps = total_steps
        self.current_step = current_step
        self.wait_steps = wait_steps
        self.logscale = logscale

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        super(TemperatureRamp, self).bind(ens, beads, nm, cell, bforce, prng, omaker)

    def step(self, step=None):
        """Updates ensemble temperature."""
        _apply_ramp(self, "temp", self.t_start, self.t_end)


class PressureRamp(Motion):
    """Pressure ramp (quench/heat)."""

    def __init__(
        self,
        fixcom=False,
        fixatoms_dof=None,
        p_start=1.0,
        p_end=1.0,
        total_steps=0,
        current_step=0,
        wait_steps=0,
        logscale=True,
    ):
        super(PressureRamp, self).__init__()
        self.p_start = p_start
        self.p_end = p_end
        self.total_steps = total_steps
        self.current_step = current_step
        self.wait_steps = wait_steps
        self.logscale = logscale

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        super(PressureRamp, self).bind(ens, beads, nm, cell, bforce, prng, omaker)

    def step(self, step=None):
        """Updates ensemble pressure."""
        _apply_ramp(self, "pext", self.p_start, self.p_end)