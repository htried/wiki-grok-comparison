#!/bin/bash
# GCP Shard Setup Script for Fixed Issues Fetcher
# This script sets up a shard of the fixed issues fetcher on a GCP VM
# Usage: ./gcp_fixed_issues_shard_setup.sh <SHARD_ID> <START_IDX> <END_IDX> [SLUGS_FILE(gs://... or path)]
# Example: ./gcp_fixed_issues_shard_setup.sh 0 0 10000

set -e

# Get parameters
SHARD_ID=${1:-0}
START_IDX=${2:-0}
END_IDX=${3:-10000}
SLUGS_FILE_ARG=${4:-}

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <SHARD_ID> <START_IDX> <END_IDX> [SLUGS_FILE(gs://... or path)]"
    echo "Example: $0 0 0 10000"
    exit 1
fi

INSTANCE_NAME="fixed-issues-shard-${SHARD_ID}"
ZONE="us-central1-a"
MACHINE_TYPE="e2-standard-4"
DISK_SIZE="200GB"
REPO_ORG="htried"
REPO_NAME="wiki-grok-comparison"

echo "Setting up fixed issues shard $SHARD_ID"
echo "Slug range: $START_IDX to $END_IDX"
[ -n "$SLUGS_FILE_ARG" ] && echo "Slugs file provided: $SLUGS_FILE_ARG"

# Check if .env file exists locally and read credentials
if [ -f ".env" ]; then
    echo "Reading credentials from .env file..."
    source .env
    # Debug: verify variables are loaded (hide sensitive values)
    if [ ! -z "$BRIGHTDATA_USERNAME" ]; then
        echo "  ✓ BRIGHTDATA_USERNAME loaded"
    fi
    if [ ! -z "$BRIGHTDATA_PASSWORD" ]; then
        echo "  ✓ BRIGHTDATA_PASSWORD loaded"
    fi
    if [ ! -z "$HF_API_TOKEN" ]; then
        echo "  ✓ HF_API_TOKEN loaded"
    fi
    if [ ! -z "$GH_USERNAME" ]; then
        echo "  ✓ GH_USERNAME loaded"
    fi
    if [ ! -z "$GH_PAT" ]; then
        echo "  ✓ GH_PAT loaded"
    fi
else
    echo "Warning: .env file not found. Make sure BRIGHTDATA_USERNAME, BRIGHTDATA_PASSWORD, HF_API_TOKEN, GH_USERNAME, and GH_PAT are set."
fi

# Create startup script
STARTUP_SCRIPT=$(cat <<EOF
#!/bin/bash
# Don't use set -e because we handle errors explicitly with || operators

# Install dependencies
apt-get update
apt-get install -y python3-pip git python3-venv

# Set HOME early (needed for git config)
export HOME=\${HOME:-/root}
mkdir -p \$HOME

# Get GitHub credentials from metadata
METADATA_GH_USERNAME=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/GH_USERNAME" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
METADATA_GH_PAT=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/GH_PAT" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
# Optional slugs file from metadata
METADATA_SLUGS_FILE=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/SLUGS_FILE" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")

# Clone repository with authentication (or pull if exists)
cd /home
if [ -d "wiki-grok-comparison" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd wiki-grok-comparison
    if [ ! -z "\$METADATA_GH_USERNAME" ] && [ ! -z "\$METADATA_GH_PAT" ]; then
        git config --global url."https://\${METADATA_GH_USERNAME}:\${METADATA_GH_PAT}@github.com/".insteadOf "https://github.com/" || echo "Warning: Git config failed"
    fi
    git pull || { echo "Git pull failed, trying fresh clone..."; cd ..; rm -rf wiki-grok-comparison; }
fi

if [ ! -d "wiki-grok-comparison" ]; then
    if [ ! -z "\$METADATA_GH_USERNAME" ] && [ ! -z "\$METADATA_GH_PAT" ]; then
        echo "Cloning private repository with credentials..."
        git clone https://\${METADATA_GH_USERNAME}:\${METADATA_GH_PAT}@github.com/${REPO_ORG}/${REPO_NAME}.git wiki-grok-comparison
        if [ \$? -ne 0 ]; then
            echo "ERROR: Git clone failed!"
            exit 1
        fi
    else
        echo "Warning: GitHub credentials not found, trying public clone (will fail if repo is private)..."
        git clone https://github.com/${REPO_ORG}/${REPO_NAME}.git wiki-grok-comparison
        if [ \$? -ne 0 ]; then
            echo "ERROR: Git clone failed!"
            exit 1
        fi
    fi
    echo "Repository cloned successfully"
fi

# Ensure we have the latest code
cd wiki-grok-comparison
echo "Verifying repository contents..."
ls -la

# Set HOME if not set (needed for git config)
export HOME=\${HOME:-/home}

echo "Pulling latest changes..."
if [ ! -z "\$METADATA_GH_USERNAME" ] && [ ! -z "\$METADATA_GH_PAT" ]; then
    git config --global url."https://\${METADATA_GH_USERNAME}:\${METADATA_GH_PAT}@github.com/".insteadOf "https://github.com/" || echo "Warning: Git config failed"
fi
git pull || echo "Warning: Git pull failed, but continuing with existing code..."
cd ..

# Store the absolute path to the repo
if [ -d "/home/wiki-grok-comparison" ]; then
    REPO_DIR=\$(cd /home/wiki-grok-comparison && pwd)
else
    REPO_DIR=\$(find /home -name "wiki-grok-comparison" -type d 2>/dev/null | head -1)
    if [ -z "\$REPO_DIR" ]; then
        echo "ERROR: Could not find repository directory!"
        exit 1
    fi
    REPO_DIR=\$(cd \$REPO_DIR && pwd)
fi
echo "Repository located at: \$REPO_DIR"
cd \$REPO_DIR

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Setup .env file from metadata (if available)
echo "# Environment variables" > .env
METADATA_BRIGHTDATA_USERNAME=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/BRIGHTDATA_USERNAME" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
METADATA_BRIGHTDATA_PASSWORD=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/BRIGHTDATA_PASSWORD" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
METADATA_HF_API_TOKEN=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/HF_API_TOKEN" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")

if [ ! -z "\$METADATA_BRIGHTDATA_USERNAME" ]; then
    echo "BRIGHTDATA_USERNAME=\$METADATA_BRIGHTDATA_USERNAME" >> .env
fi
if [ ! -z "\$METADATA_BRIGHTDATA_PASSWORD" ]; then
    echo "BRIGHTDATA_PASSWORD=\$METADATA_BRIGHTDATA_PASSWORD" >> .env
fi
if [ ! -z "\$METADATA_HF_API_TOKEN" ]; then
    echo "HF_API_TOKEN=\$METADATA_HF_API_TOKEN" >> .env
fi

# Authenticate with GCP (using default service account)
unset GOOGLE_APPLICATION_CREDENTIALS

# Make sure we're in the repo directory and activate venv
cd \$REPO_DIR || { echo "Failed to cd to repo directory"; exit 1; }
source \$REPO_DIR/venv/bin/activate

# Navigate to scripts directory and run fixed issues fetcher
cd \$REPO_DIR/scripts/gcp || { echo "Failed to cd to scripts/gcp directory"; exit 1; }
echo "Changed to scripts/gcp directory: \$(pwd)"
echo "Running fixed_issues_runner.py..."

# Prepare optional slugs file (download from GCS if provided)
EXTRA_SLUGS_FLAG=""
if [ ! -z "\$METADATA_SLUGS_FILE" ] && ! echo "\$METADATA_SLUGS_FILE" | grep -q '<!DOCTYPE\|<html\|Error 404'; then
    echo "Slugs file metadata detected: \$METADATA_SLUGS_FILE"
    if echo "\$METADATA_SLUGS_FILE" | grep -q '^gs://'; then
        echo "Downloading slugs file from GCS via python helper..."
        SLUGS_GS="\$METADATA_SLUGS_FILE" \$REPO_DIR/venv/bin/python3 - <<'PY'
import os
from google.cloud import storage

src = os.environ.get('SLUGS_GS')
dst = '/home/wiki-grok-comparison/slugs.txt'
if not src or not src.startswith('gs://'):
    raise SystemExit(f'Invalid GCS path: {src!r}')
_, path = src.split('gs://', 1)
bucket_name, blob_name = path.split('/', 1)
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_name)
blob.download_to_filename(dst)
print(f'Downloaded {src} to {dst}')
PY
        if [ -f "/home/wiki-grok-comparison/slugs.txt" ]; then
            EXTRA_SLUGS_FLAG="--slugs_file /home/wiki-grok-comparison/slugs.txt"
        else
            echo "Warning: Failed to download slugs file from GCS, continuing without it"
        fi
    else
        # Check if local file exists
        if [ -f "\$METADATA_SLUGS_FILE" ]; then
            EXTRA_SLUGS_FLAG="--slugs_file \$METADATA_SLUGS_FILE"
        else
            echo "Warning: Slugs file not found at \$METADATA_SLUGS_FILE, continuing without it"
        fi
    fi
fi

python3 \$REPO_DIR/scripts/gcp/fixed_issues_runner.py --shard_id ${SHARD_ID} --start_idx ${START_IDX} --end_idx ${END_IDX} \$EXTRA_SLUGS_FLAG 2>&1 | tee /var/log/fixed-issues-scraper.log

# Shutdown instance when done (optional - comment out if you want to keep it running)
# shutdown -h now
EOF
)

# Create VM with metadata for credentials (if .env exists)
if [ -f ".env" ]; then
    echo "Passing credentials as instance metadata..."
    
    # Build metadata string
    METADATA_STRING=""
    if [ ! -z "$BRIGHTDATA_USERNAME" ]; then
        METADATA_STRING="BRIGHTDATA_USERNAME=${BRIGHTDATA_USERNAME}"
    fi
    if [ ! -z "$BRIGHTDATA_PASSWORD" ]; then
        if [ ! -z "$METADATA_STRING" ]; then
            METADATA_STRING="${METADATA_STRING},BRIGHTDATA_PASSWORD=${BRIGHTDATA_PASSWORD}"
        else
            METADATA_STRING="BRIGHTDATA_PASSWORD=${BRIGHTDATA_PASSWORD}"
        fi
    fi
    if [ ! -z "$HF_API_TOKEN" ]; then
        if [ ! -z "$METADATA_STRING" ]; then
            METADATA_STRING="${METADATA_STRING},HF_API_TOKEN=${HF_API_TOKEN}"
        else
            METADATA_STRING="HF_API_TOKEN=${HF_API_TOKEN}"
        fi
    fi
    
    # Add GitHub credentials if they exist
    if [ ! -z "$GH_USERNAME" ] && [ ! -z "$GH_PAT" ]; then
        if [ ! -z "$METADATA_STRING" ]; then
            METADATA_STRING="${METADATA_STRING},GH_USERNAME=${GH_USERNAME},GH_PAT=${GH_PAT}"
        else
            METADATA_STRING="GH_USERNAME=${GH_USERNAME},GH_PAT=${GH_PAT}"
        fi
    fi
    
    # Build complete metadata string including SLUGS_FILE if provided
    FINAL_METADATA_STRING="$METADATA_STRING"
    if [ -n "$SLUGS_FILE_ARG" ]; then
        if [ -n "$FINAL_METADATA_STRING" ]; then
            FINAL_METADATA_STRING="${FINAL_METADATA_STRING},SLUGS_FILE=${SLUGS_FILE_ARG}"
        else
            FINAL_METADATA_STRING="SLUGS_FILE=${SLUGS_FILE_ARG}"
        fi
        echo "  Including SLUGS_FILE in metadata: ${SLUGS_FILE_ARG}"
    fi
    
    gcloud compute instances create ${INSTANCE_NAME} \
      --zone=${ZONE} \
      --machine-type=${MACHINE_TYPE} \
      --boot-disk-size=${DISK_SIZE} \
      --boot-disk-type=pd-standard \
      --tags=fixed-issues-scraper \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --metadata="${FINAL_METADATA_STRING}" \
      --metadata-from-file startup-script=<(echo "$STARTUP_SCRIPT")
else
    echo "Creating VM without credentials (set them manually or use .env file)..."
    METADATA_WITHOUT_ENV=""
    if [ -n "$SLUGS_FILE_ARG" ]; then
        METADATA_WITHOUT_ENV="SLUGS_FILE=${SLUGS_FILE_ARG}"
        echo "  Including SLUGS_FILE in metadata: ${SLUGS_FILE_ARG}"
    fi
    gcloud compute instances create ${INSTANCE_NAME} \
      --zone=${ZONE} \
      --machine-type=${MACHINE_TYPE} \
      --boot-disk-size=${DISK_SIZE} \
      --boot-disk-type=pd-standard \
      --tags=fixed-issues-scraper \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --metadata="${METADATA_WITHOUT_ENV}" \
      --metadata-from-file startup-script=<(echo "$STARTUP_SCRIPT")
fi

echo ""
echo "Instance ${INSTANCE_NAME} created"
echo "Connect with: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
echo "View logs with: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command 'tail -f /var/log/fixed-issues-scraper.log'"

