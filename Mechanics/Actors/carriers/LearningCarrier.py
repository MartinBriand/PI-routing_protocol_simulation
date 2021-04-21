"""
Learning Carrier:
We define two objects:
    * A learning agent
    * The carriers

Their behavior will be different during learning and exploiting

During learning:
    * The agent
        * Learns according to the orders of the driver. It is the extension of a TD3 tf_agents.
        * The driver will feed the agent with data from a replay buffer and ask him to learn
        * The driver will answer requests from carriers to know what to do next
    * The carriers
        * Will ask agent what to bid
        * Will generate the transition to be stored in the replay buffer
        * Will have internal parameter being changed by the driver to help the agent learn with all possible parameters
            These parameters are just the parameters of the price functions

During exploitation
    * The agent will not learn anymore
    * The driver won't change the parameters of the carriers on the way
    * The carriers won't generate episodes or these episodes won't be added to a replay buffer
"""
# TODO: is this description correct at the end of the implementation?
