#!/bin/bash


NUM_REPLICAS=4
BASE_PORT=8001
SERVER_SCRIPT=refiner_server.py

trap "trap - SIGTERM && echo 'Stopping all servers...' && kill -- -$$" SIGINT SIGTERM EXIT

for (( i=0; i<NUM_REPLICAS; i++ )); do
  CURRENT_PORT=$((BASE_PORT + i))
  GPU_ID=$((4 + i))
  
  echo "🚀 Launching VLLM replica on GPU $GPU_ID at port $CURRENT_PORT..."
  CUDA_VISIBLE_DEVICES=$GPU_ID PORT=$CURRENT_PORT python $SERVER_SCRIPT &
done

echo "✅ All $NUM_REPLICAS VLLM server replicas are starting up."
echo "   You can check their status with 'nvidia-smi'."
echo "   Endpoints are available from port $BASE_PORT to $((BASE_PORT + NUM_REPLICAS - 1))."
echo "Press Ctrl+C to shut down all servers gracefully."
wait