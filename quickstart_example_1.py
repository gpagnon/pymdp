from jax import numpy as jnp, random as jr
from pymdp import utils
from pymdp.agent import Agent

# Creates a JAX pseudorandom key, seeded with 0.
# The key (a KeyArray) is a functional RNG handle
# you can pass to jax.random ops
key = jr.PRNGKey(0)
# Split the key to derive 3 new independent subkeys and avoid
# reusing the same key (reuse gives identical draws)
keys = jr.split(key, 3)

# 2 observation modalities, 
# with 3 and 5 possible outcomes, respectively
num_obs = [3, 5]
# 2 hidden factors, 
# with 3 and 2 possible states, respectively
num_states = [3, 2]
# 2 control factors, 
# with 3 and 1 possible actions, respectively
num_controls = [3, 1]

# Generate a random likelihood array (A)
# This is a list (an ordered collection of items)
# holding two JAX arrays, one for each observation modality
A = utils.random_A_array(keys[0], num_obs, num_states)

# You can get the structure of these arrays with:
#
#   type(A)  # => list
#   len(A)   # => 2
#   print(A[0].ndim)   # => 3
#   print(A[0].shape)  # => 3, 3, 2)
#   print(A[1].ndim)   # => 3
#   print(A[1].shape)  # => (5, 3, 2)
#
# A[0][:, :, 0] is the 3x3 matrix, representing the mapping from 
#   the state factor 0 to observation modality 0,
#   when state factor 1 is equal to its first element
#
# A[0][:, :, 2] is a 3x3 matrix, representing the mapping from 
#   the state factor 0 to observation modality 0,
#   when state factor 1 is equal to its second element
#
# A[1][:, :, 0] is a 5x3 matrix, representing the mapping from 
#   the state factor 0 to observation modality 1,
#   when state factor 1 is equal to its first element
#
# A[1][:, :, 1] is a 5x3 matrix, representing the mapping from 
#   the state factor 0 to observation modality 1,
#   when state factor 1 is equal to its second element

# Generate a random transition array (B)
B = utils.random_B_array(keys[1], num_states, num_controls)

# len(B) => 2
#
# print(B[0].ndim)  # => 3
# print(B[0].shape)  # => (3, 3, 3)
#
# B[0][:, :, 0] is the transition matrix for state factor 0
#   under the control state 0 (for state factor 0)
#
# B[0][:, :, 1] is the transition matrix for state factor 0
#   under the control state 1 (for state factor 0)
#
# B[0][:, :, 2] is the transition matrix for state factor 0
#   under the control state 2 (for state factor 0)
#
# print(B[1].ndim)  # => 3
# print(B[1].shape)  # => (2, 2, 1)
#
# B[1][:, :, 0] is the transition matrix for state factor 1
#   under the control state 0 (for state factor 1)

# Generate a uniform preference array
C = utils.list_array_uniform([[no] for no in num_obs])

# len(C) => 2
#
# C[0] is the preference vector for observation modality 0
#   print(C[0].ndim)  # => 1
#   print(C[0].shape)  # => (3,)
#   print(C[0]) # => [0.33333334 0.33333334 0.33333334]
#
# C[1] is the preference vector for observation modality 1
#   print(C[1].ndim)  # => 1
#   print(C[1].shape)  # => (5,)
#   print(C[1]) # => [0.2 0.2 0.2 0.2 0.2]


agent = Agent(A=A, B=B, C=C, batch_size=1)

# Discrete observation indices for each modality
obs = [jnp.array([1]), jnp.array([2])]

# Use agent.D as the initial empirical prior
qs, info = agent.infer_states(obs, empirical_prior=agent.D, return_info=True)
# Optional diagnostic: current variational free energy for each batch element.
vfe = info["vfe"]
q_pi, neg_efe = agent.infer_policies(qs)

sample_keys = jr.split(keys[2], agent.batch_size + 1)
action = agent.sample_action(q_pi, rng_key=sample_keys[1:])