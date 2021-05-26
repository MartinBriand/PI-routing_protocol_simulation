#!/usr/bin/env python

"""
This is the learning driver for the game with learning agents.
As you may see, it is more a script than a class.

One may not that it is not a tf_agents.driver.Driver because this class is a bit inappropriate to what we do.
    We have nothing like an environment, policies are part of the LearningCarriers and observers are part of the
    LearningAgents

It is supposed to:
    * Instantiate everything
    * Run an iteration loop on the environment (During which the Carriers will share their experience
        in a replay buffer)
    * Regularly ask the learning agent to update
    * Regularly change the costs parameters of the Carriers
    * Save the model if we are satisfied
"""

from tf_agents.utils.common import function as tfa_function

from PI_RPS.Games.Learning_Game.initialize import load_env_and_agent

n_carriers_per_node = 15  # @param {type:"integer"}
discount = 0.95  # @param {type:"number"}

starting_exploration_noise = 500  # @param {type:"number"}
final_exploration_noise = 30  # @param {type:"number"}
exploration_noise = starting_exploration_noise

target_update_tau_p = 0.1  # @param {type:"number"}
target_update_period_p = 2  # @param {type:"number"}
actor_update_period_p = 2  # @param {type:"integer"}

reward_scale_factor_p = 1/500  # do not change, keep fixed at 500

target_policy_noise_p = 30.  # @param {type:"number"}
target_policy_noise_clip_p = 75.  # @param {type:"number"}

max_time_not_at_home = 30  # @param {type:"integer"}


e, learning_agent = load_env_and_agent(n_carriers=11*n_carriers_per_node,
                                       discount=discount,
                                       exploration_noise=starting_exploration_noise,
                                       target_update_tau_p=target_update_tau_p,
                                       target_update_period_p=target_update_period_p,
                                       actor_update_period_p=actor_update_period_p,
                                       reward_scale_factor_p=reward_scale_factor_p,
                                       target_policy_noise_p=target_policy_noise_p,
                                       target_policy_noise_clip_p=target_policy_noise_clip_p,
                                       max_time_not_at_home=max_time_not_at_home)

num_rounds = 30  # @param {type:"integer"}
num_cost_pass = 4  # @param {type:"integer"}
num_train_per_pass = 15  # @param {type:"integer"}

exploration_noise_update = (starting_exploration_noise - final_exploration_noise) / (num_rounds - 1)


def change_costs():
    for carrier in learning_agent.carriers:
        carrier.random_new_cost_parameters()


train = tfa_function(learning_agent.train)


# initialize the test lists to []
for i in range(num_rounds):
    print("Test", i, '/', num_rounds-1)
    change_costs()
    # run a test
    # print test results
    for j in range(num_cost_pass):
        print("Pass", j, "/", num_cost_pass-1)
        change_costs()
        for k in range(num_train_per_pass):
            print("Training", k, "/", num_train_per_pass-1)
            e.iteration()
            e.shuffle_new_transition_carriers()
            n = len(e.new_transition_carriers)
            for _ in range(n):
                carrier = e.pop_new_transition_carriers()
                experience, _ = next(carrier.training_data_set_iter)
                train(experience=experience, weights=None)
    exploration_noise -= exploration_noise_update
    learning_agent.change_exploration_noise_std(exploration_noise)
