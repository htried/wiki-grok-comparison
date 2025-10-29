#!/bin/bash
# GCP Shard Setup Script
# This script sets up a shard of the grokipedia scraper on a GCP VM
# Usage: ./gcp_shard_setup.sh <SHARD_ID> <START_IDX> <END_IDX>
# Example: ./gcp_shard_setup.sh 0 0 100000

set -e

# Get parameters
SHARD_ID=${1:-0}
START_IDX=${2:-0}
END_IDX=${3:-100000}

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <SHARD_ID> <START_IDX> <END_IDX>"
    echo "Example: $0 0 0 100000"
    exit 1
fi

INSTANCE_NAME="grokipedia-shard-${SHARD_ID}"
ZONE="us-central1-a"
MACHINE_TYPE="e2-standard-4"
DISK_SIZE="200GB"
REPO_ORG="htried"
REPO_NAME="wiki-grok-comparison"

echo "Setting up shard $SHARD_ID"
echo "URL range: $START_IDX to $END_IDX"

# Check if .env file exists locally and read credentials
if [ -f ".env" ]; then
    echo "Reading credentials from .env file..."
    source .env
else
    echo "Warning: .env file not found. Make sure to set BRIGHTDATA_USERNAME, BRIGHTDATA_PASSWORD, HF_API_TOKEN, GH_USERNAME, and GH_PAT as instance metadata or environment variables."
fi

# Create startup script
STARTUP_SCRIPT=$(cat <<EOF
#!/bin/bash
set -e

# Install dependencies
apt-get update
apt-get install -y python3-pip git python3-venv

# Get GitHub credentials from metadata
METADATA_GH_USERNAME=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/GH_USERNAME" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
METADATA_GH_PAT=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/GH_PAT" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")

# Clone repository with authentication
cd /home
if [ ! -z "\$METADATA_GH_USERNAME" ] && [ ! -z "\$METADATA_GH_PAT" ]; then
    echo "Cloning private repository with credentials..."
    git clone https://\${METADATA_GH_USERNAME}:\${METADATA_GH_PAT}@github.com/${REPO_ORG}/${REPO_NAME}.git wiki-grok-comparison
else
    echo "Warning: GitHub credentials not found, trying public clone (will fail if repo is private)..."
    git clone https://github.com/${REPO_ORG}/${REPO_NAME}.git wiki-grok-comparison
fi

# Store the absolute path to the repo - try expected location first, then search
if [ -d "/home/wiki-grok-comparison" ]; then
    REPO_DIR=\$(cd /home/wiki-grok-comparison && pwd)
elif [ -d "/home/\$(whoami)/wiki-grok-comparison" ]; then
    REPO_DIR=\$(cd /home/\$(whoami)/wiki-grok-comparison && pwd)
else
    # Try to find it
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
export GOOGLE_APPLICATION_CREDENTIALS=""

# Make sure we're in the repo directory and activate venv
cd \$REPO_DIR || { echo "Failed to cd to repo directory"; exit 1; }
echo "Current directory: \$(pwd)"

# Verify scripts directory exists
if [ ! -d "\$REPO_DIR/scripts" ]; then
    echo "ERROR: scripts directory not found in \$REPO_DIR!"
    echo "Available directories and files in \$REPO_DIR:"
    ls -la \$REPO_DIR
    exit 1
fi

# Activate virtual environment
if [ ! -f "\$REPO_DIR/venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at \$REPO_DIR/venv"
    exit 1
fi
source \$REPO_DIR/venv/bin/activate
echo "Virtual environment activated"
echo "Python path: \$(which python3)"

# Navigate to scripts directory and run scraper
cd \$REPO_DIR/scripts || { echo "Failed to cd to scripts directory"; exit 1; }
echo "Changed to scripts directory: \$(pwd)"
echo "Running shard_runner.py..."
python3 \$REPO_DIR/scripts/shard_runner.py --shard_id ${SHARD_ID} --start_idx ${START_IDX} --end_idx ${END_IDX} 2>&1 | tee /var/log/grokipedia-scraper.log

# Shutdown instance when done (optional - comment out if you want to keep it running)
# shutdown -h now
EOF
)

# Create VM with metadata for credentials (if .env exists)
if [ -f ".env" ]; then
    echo "Passing credentials as instance metadata..."
    
    # Build metadata string
    METADATA_STRING="BRIGHTDATA_USERNAME=${BRIGHTDATA_USERNAME},BRIGHTDATA_PASSWORD=${BRIGHTDATA_PASSWORD},HF_API_TOKEN=${HF_API_TOKEN}"
    
    # Add GitHub credentials if they exist
    if [ ! -z "$GH_USERNAME" ] && [ ! -z "$GH_PAT" ]; then
        METADATA_STRING="${METADATA_STRING},GH_USERNAME=${GH_USERNAME},GH_PAT=${GH_PAT}"
    fi
    
    gcloud compute instances create ${INSTANCE_NAME} \
      --zone=${ZONE} \
      --machine-type=${MACHINE_TYPE} \
      --boot-disk-size=${DISK_SIZE} \
      --boot-disk-type=pd-standard \
      --tags=grokipedia-scraper \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --metadata="${METADATA_STRING}" \
      --metadata-from-file startup-script=<(echo "$STARTUP_SCRIPT")
else
    echo "Creating VM without credentials (set them manually or use .env file)..."
    gcloud compute instances create ${INSTANCE_NAME} \
      --zone=${ZONE} \
      --machine-type=${MACHINE_TYPE} \
      --boot-disk-size=${DISK_SIZE} \
      --boot-disk-type=pd-standard \
      --tags=grokipedia-scraper \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --metadata-from-file startup-script=<(echo "$STARTUP_SCRIPT")
fi

echo ""
echo "Instance ${INSTANCE_NAME} created"
echo "Connect with: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
echo "View logs with: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command 'tail -f /var/log/grokipedia-scraper.log'"
