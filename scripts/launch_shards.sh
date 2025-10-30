#!/bin/bash
# Helper script to launch multiple GCP shard instances
# This script calculates shard boundaries and creates multiple VM instances
# Usage: ./launch_shards.sh <NUM_SHARDS> <SHARD_SIZE> [START_OFFSET] [START_FROM_SHARD] [URLS_FILE]
# Example: ./launch_shards.sh 10 88528                                  # 10 shards, 88528 URLs each, starting from 0
# Example: ./launch_shards.sh 2 10 800000                               # 2 shards, 10 URLs each, starting from index 800000
# Example: ./launch_shards.sh 4 50000 0 0 /absolute/path/to/urls.txt    # Use local newline-delimited URLs file

set -e

NUM_SHARDS=${1:-10}
SHARD_SIZE=${2:-88528}
START_OFFSET=${3:-0}
START_FROM_SHARD=${4:-0}
URLS_FILE_INPUT=${5:-}

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <NUM_SHARDS> <SHARD_SIZE> [START_OFFSET] [START_FROM_SHARD]"
    echo ""
    echo "Examples:"
    echo "  $0 10 88528                                  # 10 shards, 88528 URLs each, starting from 0"
    echo "  $0 2 10 800000                               # 2 shards, 10 URLs each, starting from index 800000"
    echo "  $0 5 1000 10000 2                            # 5 shards, 1000 URLs each, starting from 10000, but only launch shards 2-4"
    echo "  $0 4 50000 0 0 /abs/path/urls.txt            # Use local newline-delimited URLs file"
    exit 1
fi

echo "Launching ${NUM_SHARDS} shards"
echo "Shard size: ${SHARD_SIZE} URLs per shard"
echo "Starting offset: ${START_OFFSET}"
[ -n "$URLS_FILE_INPUT" ] && echo "Using URLs file: ${URLS_FILE_INPUT}"
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

# If a local URLs file is provided and not a gs:// path, upload it to GCS for VMs to access
URLS_FILE_TO_PASS=""
if [ -n "$URLS_FILE_INPUT" ]; then
    if [[ "$URLS_FILE_INPUT" == gs://* ]]; then
        URLS_FILE_TO_PASS="$URLS_FILE_INPUT"
    else
        # Determine staging bucket (use existing output bucket by default or override via STAGING_BUCKET env var)
        STAGING_BUCKET=${STAGING_BUCKET:-enwiki-structured-contents-20251028}
        BASENAME=$(basename "$URLS_FILE_INPUT")
        GS_PATH="gs://${STAGING_BUCKET}/shard_inputs/${BASENAME}"
        echo "Uploading local URLs file to ${GS_PATH}..."
        gsutil cp "$URLS_FILE_INPUT" "$GS_PATH"
        URLS_FILE_TO_PASS="$GS_PATH"
    fi
fi

# Launch each shard
for ((SHARD_ID=$START_FROM_SHARD; SHARD_ID<$NUM_SHARDS; SHARD_ID++)); do
    START_IDX=$((START_OFFSET + SHARD_ID * SHARD_SIZE))
    END_IDX=$((START_IDX + SHARD_SIZE))
    
    echo "Launching shard ${SHARD_ID}: URLs ${START_IDX} to ${END_IDX} ($((END_IDX - START_IDX)) URLs)"
    
    # Call the setup script
    if [ -n "$URLS_FILE_TO_PASS" ]; then
        ./scripts/gcp_shard_setup.sh ${SHARD_ID} ${START_IDX} ${END_IDX} "$URLS_FILE_TO_PASS"
    else
        ./scripts/gcp_shard_setup.sh ${SHARD_ID} ${START_IDX} ${END_IDX}
    fi
    
    echo "Waiting 5 seconds before launching next shard..."
    sleep 5
done

echo ""
echo "All shards launched!"
echo "Monitor with: gcloud compute instances list --filter='tags.items:grokipedia-scraper'"
echo ""
echo "To check logs for a specific shard:"
echo "  gcloud compute ssh grokipedia-shard-<ID> --zone=us-central1-a --command 'tail -f /var/log/grokipedia-scraper.log'"

