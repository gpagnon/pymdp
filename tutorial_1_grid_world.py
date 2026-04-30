## Clean code extracted from the tutorial_1_grid_world.ipynb notebook

# Imports PyTree utilities (tree_map)
import jax.tree_util as jtu
# Imports JAX NumPy API
from jax import numpy as jnp
# JAX random module (not used here)
from jax import random as jr
# Agent class for inference/action selection
from pymdp.agent import Agent
# Build model tensors from a structured spec
from pymdp.distribution import compile_model
# Environment base class
from pymdp.envs.env import Env
# Rollout helper (not used here)
from pymdp.envs import rollout

# Define the labels for the agent's position on the 1D grid
# (both states and observation categories)
positions = ["left", "center_left", "center_right", "right"]
# Define the labels for the agent's available actions
# (control categories)
actions = ["move_left", "move_right"]

# The model description is specified by a nested dictionary, that contains:
# - observations: just 1 modality (observing the agent's position)
# - controls: just 1 control factor (the agent's movement)
# - states: just 1 hidden state (the agent's position)
#
# Note that the position observation "depends on" the agent's position:
# this specifies the dependency structure of the A matrix
#
# Similarly, the position hidden factor depends on the agent's (previous)
# position: this specifies the dependency structure of the B matrix. 
# Furthermore, this state factor is "controlled by" the movement control state.
#
# Nested speicification of the generative model:
model_description = {
    "observations": {
        "position_obs": {
            "elements": positions, 
            "depends_on": ["position"] # we specify that the observation depends on the "position" state factor
        },
    },
    "controls": {
        "movement": {
            "elements": actions} # we specify the available actions
    },
    "states": {
        "position": {
            "elements": positions, 
            "depends_on": ["position"],  # our current position depends on previous position...
            "controlled_by": ["movement"]  # ...and the movement action taken
        },
    },
}

# Compile the model structure from the description:
# create empty labeled tensors (A, B, C, D, etc.) consistent with the specs
model = compile_model(model_description)

# Fill in the likelihood (A) tensor
# The observations have an identical mapping to the states 
# (i.e., the agent will perfectly observe its position = "perfect sensing")
model.A["position_obs"]["left", "left"] = 1.0
model.A["position_obs"]["center_left", "center_left"] = 1.0
model.A["position_obs"]["center_right", "center_right"] = 1.0
model.A["position_obs"]["right", "right"] = 1.0
# You could also use the .data attribute to set the identity mapping directly:
# model.A["position_obs"].data = jnp.eye(len(positions)) 

# Fill in the transition model (B) tensor
# note that it's specified as ["to", "from", "action"]

# moving right
model.B["position"]["center_left", "left", "move_right"] = 1.0     
model.B["position"]["center_right", "center_left", "move_right"] = 1.0  
model.B["position"]["right", "center_right", "move_right"] = 1.0    
model.B["position"]["right", "right", "move_right"] = 1.0           

# moving left  
model.B["position"]["left", "left", "move_left"] = 1.0              
model.B["position"]["left", "center_left", "move_left"] = 1.0       
model.B["position"]["center_left", "center_right", "move_left"] = 1.0  
model.B["position"]["center_right", "right", "move_left"] = 1.0    

# set preferences (C) tensor - prefer to be at "center_left"
model.C["position_obs"]["center_left"] = 1.0

# The parameter gamma defines the degree of stocasticity in behavior.
# We choose here deterministic (i.e. highly precise) behavior; 
# make gamma smaller for stochastic behavior
gamma = 10 

# Create agent from model tensors plus gamma (the ** syntax means: 
# take all key-value pairs from model and pass them as named arguments 
# to Agent(), plus gamma=gamma)
agent = Agent(**model, gamma = gamma)

# Set up initial observation to be "left" (category index 0)
# broadcast to agent's batch size (defaults to 1 agent) and add a time dimension,
observation = jnp.zeros((agent.batch_size, 1)) 

# Get the prior:
# qs needs a time dimension too: jtu.tree_map applies a function 
# (in this case, adds a new dimension at axis 1) to every leaf
# of agent.D, that is, to all state factors; then qs_init inherits 
# the same nested structure
qs_init = jtu.tree_map(lambda x: jnp.expand_dims(x, 1), agent.D) 

# Posterior beliefs over hidden states given observations
qs = agent.infer_states([observation], qs_init)
# qs[0][0] is the belief distribution over positions for the first (and only)
# agent; displays the most probable inferred position
print(f"Current belief about position: {positions[jnp.argmax(qs[0][0])]}")

# Removes the singleton time dimension (axis 1) from each array in qs, 
# to get beliefs into the shape expected by infer_policies(qs)
qs = [jnp.squeeze(q, 1) for q in qs]

# Print goal (preferred position)
print(f"Goal position: {positions[jnp.argmax(agent.C[0])]}")

# Compute posterior over policies and expected free energies
q_pi, G = agent.infer_policies(qs)
# Samples action from policy posterior
action_idx = agent.sample_action(q_pi)
# Displays chosen action label
print(f"Action chosen: {actions[action_idx[0][0]]}")

# ------------------------------------

# Now we are going to run 3 independent agents/trials in parallel
batch_size = 3
# deterministic behavior retained
gamma = 10     

# Create agent
agent = Agent(**model, batch_size = batch_size, gamma = gamma)

# Set up different initial observations for each agent: 
# "left", "center_right", and "right".
# Wrap in a list to indicate the single modality; 
# observation[0].shape = (batch_size, 1)
observation = [jnp.array([[0], [2], [3]])] 

# Add time axis to priors as before
qs_init = jtu.tree_map(lambda x: jnp.expand_dims(x, 1), agent.D) 

# State inference for all 3 agents in one call
qs = agent.infer_states(observation, qs_init)
# Prints each agent's inferred current position
for a in range(batch_size): 
    print(f"Agent {a}'s current belief about position: {positions[jnp.argmax(qs[0][a])]}")

# Removes time axis for policy inference
qs = [jnp.squeeze(q, 1) for q in qs]

# Prints common goal (center_left)
print(f"\nGoal position for all agents: {positions[jnp.argmax(agent.C[0])]}\n")

# Computes policy beliefs / expected free energies for each agent
q_pi, G = agent.infer_policies(qs)
# Samples an action per agent
action = agent.sample_action(q_pi)
# Prints each agent's selected action
for a in range(batch_size): 
    print(f"Agent {a}'s action chosen: {actions[action[a][0]]}")