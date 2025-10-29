#!/bin/bash
# Helper script to launch multiple GCP shard instances
# This script calculates shard boundaries and creates multiple VM instances
# Usage: ./launch_shards.sh <NUM_SHARDS> <SHARD_SIZE> [START_OFFSET] [START_FROM_SHARD]
# Example: ./launch_shards.sh 10 88528  # 10 shards, 88528 URLs each, starting from 0
# Example: ./launch_shards.sh 2 10 800000  # 2 shards, 10 URLs each, starting from index 800000

set -e

NUM_SHARDS=${1:-10}
SHARD_SIZE=${2:-88528}
START_OFFSET=${3:-0}
START_FROM_SHARD=${4:-0}

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <NUM_SHARDS> <SHARD_SIZE> [START_OFFSET] [START_FROM_SHARD]"
    echo ""
    echo "Examples:"
    echo "  $0 10 88528                    # 10 shards, 88528 URLs each, starting from 0"
    echo "  $0 2 10 800000                 # 2 shards, 10 URLs each, starting from index 800000"
    echo "  $0 5 1000 10000 2              # 5 shards, 1000 URLs each, starting from 10000, but only launch shards 2-4"
    exit 1
fi

echo "Launching ${NUM_SHARDS} shards"
echo "Shard size: ${SHARD_SIZE} URLs per shard"
echo "Starting offset: ${START_OFFSET}"
echo ""

# Check if .env exists (try both relative paths)
if [ ! -f ".env" ] && [ ! -f "../.env" ]; then
    echo "Warning: .env file not found. Make sure BRIGHTDATA_USERNAME, BRIGHTDATA_PASSWORD, and HF_API_TOKEN are set."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Launch each shard
for ((SHARD_ID=$START_FROM_SHARD; SHARD_ID<$NUM_SHARDS; SHARD_ID++)); do
    START_IDX=$((START_OFFSET + SHARD_ID * SHARD_SIZE))
    END_IDX=$((START_IDX + SHARD_SIZE))
    
    echo "Launching shard ${SHARD_ID}: URLs ${START_IDX} to ${END_IDX} ($((END_IDX - START_IDX)) URLs)"
    
    # Call the setup script
    ./scripts/gcp_shard_setup.sh ${SHARD_ID} ${START_IDX} ${END_IDX}
    
    echo "Waiting 5 seconds before launching next shard..."
    sleep 5
done

echo ""
echo "All shards launched!"
echo "Monitor with: gcloud compute instances list --filter='tags.items:grokipedia-scraper'"
echo ""
echo "To check logs for a specific shard:"
echo "  gcloud compute ssh grokipedia-shard-<ID> --zone=us-central1-a --command 'tail -f /var/log/grokipedia-scraper.log'"

