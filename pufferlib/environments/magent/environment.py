"""Construct the original PettingZoo MAGent SDK through its legacy API."""

import functools

import pufferlib
import pufferlib.emulation


def env_creator(name="battle_v4"):
    """Select a MAGent environment factory."""
    return functools.partial(make, name)


def make(name, buf=None, seed=0, **kwargs):
    """Create the SDK's native parallel environment with its supported options."""
    if name != "battle_v4":
        raise ValueError(f"Unknown environment name {name}")
    from pettingzoo.magent import battle_v4

    env = MagentWrapper(battle_v4.parallel_env(**kwargs))
    return pufferlib.emulation.PettingZooPufferEnv(env=env, buf=buf, seed=seed)


class MagentWrapper(pufferlib.PettingZooWrapper):
    """Preserve legacy done flags while exposing the five-value parallel API."""

    def reset(self, seed=None, options=None):
        """Reset the original SDK and provide its missing empty info mapping."""
        observations = self.env.reset(seed=seed, options=options)
        return observations, {agent: {} for agent in observations}

    def step(self, actions):
        """Keep SDK done flags as terminations without inventing lost truncations."""
        observations, rewards, dones, infos = self.env.step(actions)
        truncations = {agent: False for agent in dones}
        return observations, rewards, dones, truncations, infos
