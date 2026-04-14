from jax import jit
from jax import random as jr
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs.env import make
from pymdp.envs.rollout import rollout

key = jr.PRNGKey(1)
key_A, key_B, key_rollout = jr.split(key, 3)

num_obs = [3, 5]
num_states = [3, 2]
num_controls = [3, 1]

A = utils.random_A_array(key_A, num_obs, num_states)
B = utils.random_B_array(key_B, num_states, num_controls)
C = utils.list_array_uniform([[no] for no in num_obs])
D = utils.list_array_uniform([[ns] for ns in num_states])

agent = Agent(A=A, B=B, C=C, D=D, batch_size=1)
env, _ = make(A=A, B=B, D=D)

rollout_jit = jit(rollout, static_argnums=[1, 2])  # env and num_timesteps are static
last, info = rollout_jit(
    agent,
    env,
    20,
    key_rollout,
)

actions = info["action"]