#!/bin/bash
# Helper script to launch multiple GCP shard instances for fixed issues fetching
# Usage: ./launch_fixed_issues_shards.sh <NUM_SHARDS> [SHARD_SIZE] [START_OFFSET] [START_FROM_SHARD] [SLUGS_FILE]
# Example: ./launch_fixed_issues_shards.sh 10 10000                    # 10 shards, 10k slugs each (default), starting from 0
# Example: ./launch_fixed_issues_shards.sh 2 10000 0 0 gs://bucket/slugs.txt  # Use GCS slugs file

set -e

NUM_SHARDS=${1:-10}
SHARD_SIZE=${2:-10000}  # Default to 10k per shard to avoid blocking
START_OFFSET=${3:-0}
START_FROM_SHARD=${4:-0}
# Default slugs file (relative to scripts/gcp directory, or use absolute path)
DEFAULT_SLUGS_FILE="results/overall/grokipedia_w_license.txt"
SLUGS_FILE_INPUT=${5:-$DEFAULT_SLUGS_FILE}

if [ -z "$1" ]; then
    echo "Usage: $0 <NUM_SHARDS> [SHARD_SIZE] [START_OFFSET] [START_FROM_SHARD] [SLUGS_FILE]"
    echo ""
    echo "Examples:"
    echo "  $0 10 10000                                  # 10 shards, 10k slugs each (default), starting from 0"
    echo "  $0 2 10000 0 0                                # 2 shards, 10k slugs each, starting from 0"
    echo "  $0 5 10000 10000 2                            # 5 shards, 10k slugs each, starting from 10000, but only launch shards 2-4"
    echo "  $0 4 10000 0 0 gs://bucket/slugs.txt          # Use GCS slugs file"
    echo "  $0 4 10000 0 0 /abs/path/slugs.txt           # Use local slugs file (will be uploaded to GCS)"
    exit 1
fi

echo "Launching ${NUM_SHARDS} shards for fixed issues fetching"
echo "Shard size: ${SHARD_SIZE} slugs per shard (default: 10k to avoid blocking)"
echo "Starting offset: ${START_OFFSET}"
[ -n "$SLUGS_FILE_INPUT" ] && echo "Using slugs file: ${SLUGS_FILE_INPUT}"
echo ""

# Check if .env exists (try both relative paths)
if [ ! -f ".env" ] && [ ! -f "../.env" ] && [ ! -f "../../.env" ]; then
    echo "Warning: .env file not found. Make sure BRIGHTDATA_USERNAME, BRIGHTDATA_PASSWORD, and HF_API_TOKEN are set."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# If a local slugs file is provided and not a gs:// path, upload it to GCS for VMs to access
SLUGS_FILE_TO_PASS=""
if [ -n "$SLUGS_FILE_INPUT" ]; then
    if [[ "$SLUGS_FILE_INPUT" == gs://* ]]; then
        SLUGS_FILE_TO_PASS="$SLUGS_FILE_INPUT"
    else
        # Resolve relative paths to absolute
        if [[ "$SLUGS_FILE_INPUT" != /* ]]; then
            # Relative path - make it absolute from current directory
            SLUGS_FILE_INPUT="$(cd "$(dirname "$SLUGS_FILE_INPUT")" && pwd)/$(basename "$SLUGS_FILE_INPUT")"
        fi
        
        # Check if file exists
        if [ ! -f "$SLUGS_FILE_INPUT" ]; then
            echo "Error: Slugs file not found: $SLUGS_FILE_INPUT"
            exit 1
        fi
        
        # Determine staging bucket (use existing output bucket by default or override via STAGING_BUCKET env var)
        STAGING_BUCKET=${STAGING_BUCKET:-enwiki-structured-contents-20251028}
        BASENAME=$(basename "$SLUGS_FILE_INPUT")
        GS_PATH="gs://${STAGING_BUCKET}/shard_inputs/${BASENAME}"
        echo "Uploading local slugs file to ${GS_PATH}..."
        gsutil cp "$SLUGS_FILE_INPUT" "$GS_PATH"
        SLUGS_FILE_TO_PASS="$GS_PATH"
    fi
fi

# Launch each shard
for ((SHARD_ID=$START_FROM_SHARD; SHARD_ID<$NUM_SHARDS; SHARD_ID++)); do
    START_IDX=$((START_OFFSET + SHARD_ID * SHARD_SIZE))
    END_IDX=$((START_IDX + SHARD_SIZE))
    
    echo "Launching shard ${SHARD_ID}: Slugs ${START_IDX} to ${END_IDX} ($((END_IDX - START_IDX)) slugs)"
    
    # Call the setup script
    if [ -n "$SLUGS_FILE_TO_PASS" ]; then
        ./scripts/gcp/gcp_fixed_issues_shard_setup.sh ${SHARD_ID} ${START_IDX} ${END_IDX} "$SLUGS_FILE_TO_PASS"
    else
        ./scripts/gcp/gcp_fixed_issues_shard_setup.sh ${SHARD_ID} ${START_IDX} ${END_IDX}
    fi
    
    echo "Waiting 5 seconds before launching next shard..."
    sleep 5
done

echo ""
echo "All shards launched!"
echo "Monitor with: gcloud compute instances list --filter='tags.items:fixed-issues-scraper'"
echo ""
echo "To check logs for a specific shard:"
echo "  gcloud compute ssh fixed-issues-shard-<ID> --zone=us-central1-a --command 'tail -f /var/log/fixed-issues-scraper.log'"

