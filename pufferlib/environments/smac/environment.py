import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator(name="smac"):
    """Select the SMAC factory with its original default StarCraft map."""
    return functools.partial(make, name)


def make(name, buf=None):
    """Create SMAC with its source default map and native action masks."""
    pufferlib.environments.try_import("smac")
    from smac.env.pettingzoo.StarCraft2PZEnv import _parallel_env

    env = SMAC(_parallel_env(1000))
    return pufferlib.emulation.PettingZooPufferEnv(env, buf=buf)


class SMAC(pufferlib.PettingZooWrapper):
    """Add reset infos to the SDK's observation-only ParallelEnv reset."""

    def reset(self, seed=None, options=None):
        """Return reset observations and empty infos for every current agent."""
        observations = self.env.reset(seed=seed, options=options)
        return observations, {agent: {} for agent in observations}
