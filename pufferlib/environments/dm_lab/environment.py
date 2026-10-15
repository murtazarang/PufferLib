import functools

import gymnasium as gym
import numpy as np
from shimmy import DmLabCompatibilityV0

import pufferlib.emulation
import pufferlib.environments


class IntegerDmLabActions(gym.ActionWrapper):
    """Represent every bounded native integer control vector with categorical indices."""

    def __init__(self, env, action_specification):
        """Derive integer lows, axis cardinalities, and names from the real SDK."""
        super().__init__(env)
        self.action_names = tuple(specification["name"] for specification in action_specification)
        self.action_low = np.asarray(
            [specification["min"] for specification in action_specification], dtype=np.intc
        )
        action_high = np.asarray(
            [specification["max"] for specification in action_specification], dtype=np.intc
        )
        self.action_space = gym.spaces.MultiDiscrete(action_high - self.action_low + 1)

    def action(self, action):
        """Restore native signed controls in the exact order consumed by Shimmy."""
        native_action = np.asarray(action, dtype=np.intc) + self.action_low
        return {
            name: np.asarray([value], dtype=self.env.action_space[name].dtype)
            for name, value in zip(self.action_names, native_action, strict=True)
        }


class TerminalDmLabCompatibility(DmLabCompatibilityV0):
    """Retain the last valid native frame when Lab stops before exposing observations."""

    def reset(self, *, seed=None, options=None):
        """Cache owned observation copies separately from the returned reset data."""
        observation, info = super().reset(seed=seed, options=options)
        self.last_observation = {name: value.copy() for name, value in observation.items()}
        return observation, info

    def step(self, action):
        """Preserve native rewards and flags while using the SDK's cached terminal convention."""
        action_array = np.asarray([value[0] for value in action.values()], dtype=np.intc)
        reward = self._env.step(action_array)
        terminated = not self._env.is_running()
        if terminated:
            observation = {name: value.copy() for name, value in self.last_observation.items()}
        else:
            observation = self._env.observations()
            self.last_observation = {name: value.copy() for name, value in observation.items()}
        return observation, reward, terminated, False, {}


def env_creator(name="seekavoid_arena_01"):
    """Select the original DeepMind Lab level with all native integer controls."""
    return functools.partial(make, name=name)


def make(name, buf=None):
    """Preserve native controls and return owned cached RGB data at termination."""
    dm_lab = pufferlib.environments.try_import("deepmind_lab", "dm-lab")
    raw = dm_lab.Lab(name, ["RGB_INTERLEAVED"])
    action_specification = raw.action_spec()
    env = TerminalDmLabCompatibility(env=raw)
    env = IntegerDmLabActions(env, action_specification)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
