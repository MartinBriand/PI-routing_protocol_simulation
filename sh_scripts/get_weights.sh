# this program is to execute with source (which is why it has not execute rights). Is is important to use the python environement and to attach the jobs to the session

original_path=`pwd`

# Put the directory of the present script here
cd ~/pi

# Split the sequence in smaller pieces if you don't have enough memory
for node_auction_cost in `seq 0 10 250`
do
    # Correct the path of the script and of the output files
    python PI-routing_protocol_simulation/Scripts/training_learning_node_game.py node_auction_cost=$node_auction_cost > ../last_script/get_weights/$node_auction_cost\.log 2>&1 &
done
cd $original_path
