import functools

import pettingzoo

import pufferlib.emulation
import pufferlib.environments


def env_creator(name='cooperative_pong_v6'):
    return functools.partial(make, name)


def make(name, buf=None, seed=0):
    """Create a supported Butterfly parallel environment."""
    names = {
        'cooperative_pong_v6': 'butterfly/cooperative_pong-v6',
        'knights_archers_zombies_v11': 'butterfly/knights_archers_zombies-v11',
    }
    if name not in names:
        raise ValueError(f'Unknown environment: {name}')
    pufferlib.environments.try_import('pettingzoo.butterfly', 'butterfly')
    env = pettingzoo.make('parallel', names[name])
    return pufferlib.emulation.PettingZooPufferEnv(env=env, buf=buf, seed=seed)
