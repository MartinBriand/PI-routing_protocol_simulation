"""
This is the learning driver for the game with learning agents.
It is supposed to:
    * Instantiate everything
    * Run an iteration loop on the environment (During which the carriers will share their experience
        in a replay buffer)
    * Regularly ask the learning agent to update
    * Regularly change the costs parameters of the carriers
    * Save the model if we are satisfied
"""

import tf_agents as tfa
