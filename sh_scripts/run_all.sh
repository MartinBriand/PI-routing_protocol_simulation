# This program is to execute after gettting the weights with the other program.
# this program is to execute with source (which is why it has not execute rights). Is is important to use the python environement and to attach the jobs to the session

original_path=`pwd`

# put here the path of the present script
cd ~/pi

# Split the loops in smaller pieces if you don't have enough memory
for auction_type in 'MultiLanes' 'SingleLane'
do
    for node_auction_cost in `seq 0 10 250`
    do
        for terminaison_file_name in `seq 1 3`
        do
            # Correct the path of the script as well as the folder where you want to save the output files
            python PI-routing_protocol_simulation/Scripts/all_convergence.py auction_type=$auction_type  node_auction_cost=$node_auction_cost\. terminaison_file_name=$terminaison_file_name > ../last_script/runs/$auction_type\_$node_auction_cost\.0_$terminaison_file_name\.log 2>&1 &
        done    
    done
done
cd $original_path
