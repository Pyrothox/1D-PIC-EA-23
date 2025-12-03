#!/bin/bash

start_sim () {
    local folder=$1
    local output_file="$folder/out.txt"
    python main.py -r "$folder" > "$output_file" 2>&1 &
    local pid=$!
    echo "Simulation started with PID $pid"
    trap "kill -SIGINT $pid && echo 'Simulation cancelled'" INT
}