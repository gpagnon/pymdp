# quickstart_example_2.py
#
# Run a quick agent-environment loop with rollout() and PymdpEnv. 
# For repeated calls, we recommend wrapping rollout in jit

from jax import jit
from jax import random as jr
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs.env import make
from pymdp.envs.rollout import rollout

# Creates a JAX pseudorandom key, seeded with 0.
# The key (a KeyArray) is a functional RNG handle
# you can pass to jax.random ops
key = jr.PRNGKey(1)
# Split the key to derive 3 new independent subkeys and avoid
# reusing the same key (reuse gives identical draws)
key_A, key_B, key_rollout = jr.split(key, 3)

# 2 observation modalities, 
# with 3 and 5 possible outcomes, respectively
num_obs = [3, 5]
# 2 hidden factors, 
# with 3 and 2 possible states, respectively
num_states = [3, 2]
# 2 control factors, 
# with 3 and 1 possible actions, respectively
num_controls = [3, 1]

# Generate random model parameter arrays
A = utils.random_A_array(key_A, num_obs, num_states)
B = utils.random_B_array(key_B, num_states, num_controls)
C = utils.list_array_uniform([[no] for no in num_obs])
D = utils.list_array_uniform([[ns] for ns in num_states])

# Build an agent based on the above model parameters
agent = Agent(A=A, B=B, C=C, D=D, batch_size=1)
# Build an environment (generative process?)
env, _ = make(A=A, B=B, D=D)

# JIT-compiles the function rollout(), specifying that 
# positional arguments 1 (env) and 2 (num_timesteps) 
# are static (ie, compile-time constants).
rollout_jit = jit(rollout, static_argnums=[1, 2])  

# Calls the rollout function, which runs 20 timesteps
last, info = rollout_jit(
    agent,
    env,
    20,
    key_rollout,
)

actions = info["action"]