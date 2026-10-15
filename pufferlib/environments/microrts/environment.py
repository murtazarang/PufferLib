import functools

import gym
import numpy as np
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator(name="GlobalAgentCombinedRewardEnv"):
    return functools.partial(make, name)


def make(name, buf=None):
    """Create the source-pinned MicroRTS SDK through Gymnasium emulation."""
    pufferlib.environments.try_import("gym_microrts")
    if name == "GlobalAgentCombinedRewardEnv":
        from gym_microrts.envs import GlobalAgentCombinedRewardEnv
    else:
        raise ValueError(f"Unknown environment: {name}")

    with pufferlib.Suppress():
        env = GlobalAgentCombinedRewardEnv()

    env.reset = pufferlib.silence_warnings(env.reset)
    env.step = pufferlib.silence_warnings(env.step)
    env = MicroRTS(env)
    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


class MicroRTS(gym.Wrapper):
    """Match the SDK observations to its declared int32 observation space."""

    def reset(self):
        return self.env.reset().astype(np.int32)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation.astype(np.int32), reward, done, info
