# GCP Deployment Scripts

Scripts for deploying the Grokipedia scraper on Google Cloud Platform using sharded Compute Engine instances.

## Overview

This directory contains scripts to:
1. Set up individual GCP VM instances for scraping a shard of URLs
2. Launch multiple shards in parallel
3. Run the scraper on each VM with proper configuration
4. Upload results directly to Google Cloud Storage (GCS)

## Files

- **`grokipedia_scraper.py`** - Core scraping logic (parsing, rate limiting, GCS upload)
- **`shard_runner.py`** - Entry point script that runs on each VM
- **`gcp_shard_setup.sh`** - Creates a single GCP VM instance for one shard
- **`launch_shards.sh`** - Orchestrates launching multiple shard instances

## Prerequisites

1. **GCP Setup:**
   - Google Cloud SDK (`gcloud`) installed and authenticated
   - Project set: `gcloud config set project vitaly-gcp`
   - Default service account with GCS write permissions
   - Billing enabled

2. **Local `.env` file** (in project root) with credentials:
   ```
   BRIGHTDATA_USERNAME=your_username
   BRIGHTDATA_PASSWORD=your_password
   HF_API_TOKEN=your_huggingface_token
   GH_USERNAME=your_github_username
   GH_PAT=your_github_personal_access_token
   ```

3. **GitHub Repository:**
   - Private repository: `https://github.com/htried/wiki-grok-comparison`
   - GitHub PAT with repo access

## Quick Start

### Launch Multiple Shards

The easiest way to launch multiple shards:

```bash
cd scripts/gcp

# Launch 10 shards, 88528 URLs each, starting from index 0
./launch_shards.sh 10 88528

# Launch 2 shards for testing, 10 URLs each, starting from index 800000
./launch_shards.sh 2 10 800000

# Use a local URLs file (will be uploaded to GCS)
./launch_shards.sh 4 50000 0 0 /absolute/path/to/urls.txt
```

### Create a Single Shard Instance

To create just one VM instance:

```bash
cd scripts/gcp

# Create shard 0 processing URLs 0-100000
./gcp_shard_setup.sh 0 0 100000

# With a local URLs file
./gcp_shard_setup.sh 0 0 100000 /absolute/path/to/urls.txt

# With a GCS URLs file
./gcp_shard_setup.sh 0 0 100000 gs://bucket-name/urls.txt
```

## Script Details

### `launch_shards.sh`

Orchestrates launching multiple GCP VM instances in parallel.

**Usage:**
```bash
./launch_shards.sh <NUM_SHARDS> <SHARD_SIZE> [START_OFFSET] [START_FROM_SHARD] [URLS_FILE]
```

**Arguments:**
- `NUM_SHARDS`: Number of shard instances to create
- `SHARD_SIZE`: Number of URLs per shard
- `START_OFFSET`: Starting URL index (default: 0)
- `START_FROM_SHARD`: Which shard number to start from (default: 0)
- `URLS_FILE`: Optional local file or GCS path (`gs://...`) to newline-delimited URLs

**Examples:**
```bash
# 10 shards, 88528 URLs each, starting from 0
./launch_shards.sh 10 88528

# 2 shards, 10 URLs each, starting from index 800000
./launch_shards.sh 2 10 800000

# Launch shards 2-4 only (5 shards total, but only create 2,3,4)
./launch_shards.sh 5 1000 10000 2

# Use local URLs file (will be uploaded to GCS)
./launch_shards.sh 4 50000 0 0 /abs/path/to/urls.txt
```

**Behavior:**
- Calculates start/end indices for each shard
- If `URLS_FILE` is local, uploads it to GCS first
- Calls `gcp_shard_setup.sh` for each shard with a 5-second delay
- Reads credentials from `.env` file (checks multiple locations)

### `gcp_shard_setup.sh`

Creates a single GCP Compute Engine instance for one shard.

**Usage:**
```bash
./gcp_shard_setup.sh <SHARD_ID> <START_IDX> <END_IDX> [URLS_FILE]
```

**Arguments:**
- `SHARD_ID`: Unique identifier for this shard (used in instance name)
- `START_IDX`: Start index (inclusive) for URL range
- `END_IDX`: End index (exclusive) for URL range
- `URLS_FILE`: Optional GCS path (`gs://...`) or local file path to URLs

**VM Configuration:**
- **Instance Name**: `grokipedia-shard-{SHARD_ID}`
- **Zone**: `us-central1-a`
- **Machine Type**: `e2-standard-4` (4 vCPUs, 16 GB RAM)
- **Disk Size**: 200 GB
- **OS**: Ubuntu (default Compute Engine image)

**Startup Script:**
The VM startup script automatically:
1. Installs Python 3, pip, git, venv
2. Clones the private GitHub repository using PAT from metadata
3. Creates and activates a Python virtual environment
4. Installs dependencies from `requirements.txt`
5. Creates `.env` file from instance metadata (BrightData, HuggingFace credentials)
6. Downloads URLs file from GCS if needed
7. Runs `shard_runner.py` with the specified range

**Instance Metadata:**
The script passes these as instance metadata:
- `BRIGHTDATA_USERNAME`
- `BRIGHTDATA_PASSWORD`
- `HF_API_TOKEN`
- `GH_USERNAME`
- `GH_PAT`
- `URLS_FILE` (if provided)

### `shard_runner.py`

Entry point script that runs on each VM instance.

**Usage:**
```bash
python shard_runner.py --start_idx 0 --end_idx 100000 [--shard_id 0] [--urls_file path]
```

**Arguments:**
- `--start_idx`: Start index (inclusive) for URL range
- `--end_idx`: End index (exclusive) for URL range
- `--shard_id`: Optional shard ID for logging/naming
- `--urls_file`: Optional local file path (JSONL/JSON/TXT) or GCS path

**Behavior:**
- Loads URLs from HuggingFace dataset (default) or local file
- Validates index ranges
- Calls `grokipedia_scraper.scraping_phase()` with GCS configuration
- Logs to `/var/log/grokipedia-scraper.log`

### `grokipedia_scraper.py`

Core scraping module with parsing, rate limiting, and GCS upload.

**Key Functions:**
- `load_urls_from_hf()` - Load URLs from HuggingFace dataset
- `load_urls_from_file()` - Load URLs from local file (JSONL/JSON/TXT)
- `scraping_phase()` - Main scraping orchestration with:
  - Concurrent requests with rate limiting
  - Exponential backoff retry logic
  - Incremental batch saving to GCS
  - Progress tracking

**Rate Limiting:**
- Default: 100 requests/second with 50 concurrent workers
- Uses BrightData proxy for higher throughput

## Workflow

### Typical Deployment Process

1. **Prepare URLs (optional):**
   ```bash
   # If using a local URLs file, it will be auto-uploaded to GCS
   # Otherwise, uses HuggingFace dataset by default
   ```

2. **Launch Shards:**
   ```bash
   cd scripts/gcp
   ./launch_shards.sh 10 88528
   ```

3. **Monitor Instances:**
   ```bash
   # List instances
   gcloud compute instances list --filter="name:grokipedia-shard-*"
   
   # View logs on a specific instance
   gcloud compute instances get-serial-port-output grokipedia-shard-0
   
   # SSH into instance (for debugging)
   gcloud compute ssh grokipedia-shard-0
   ```

4. **Check GCS Output:**
   ```bash
   # List uploaded files
   gsutil ls gs://enwiki-structured-contents-20251028/batch_*.jsonl
   
   # Count files
   gsutil ls gs://enwiki-structured-contents-20251028/batch_*.jsonl | wc -l
   
   # Download a sample
   gsutil cp gs://enwiki-structured-contents-20251028/batch_0_88527.jsonl .
   ```

5. **Clean Up (when done):**
   ```bash
   # Delete all shard instances
   gcloud compute instances delete grokipedia-shard-0 grokipedia-shard-1 ...
   
   # Or delete in bulk
   gcloud compute instances list --filter="name:grokipedia-shard-*" --format="value(name)" | xargs gcloud compute instances delete
   ```

## Configuration

### GCS Bucket

Default bucket is `enwiki-structured-contents-20251028` in project `vitaly-gcp`. To change:
- Edit `shard_runner.py` (lines with `gcs_bucket` and `gcs_project`)
- Or pass as environment variables on the VM

### VM Settings

To change VM configuration, edit `gcp_shard_setup.sh`:
- **Zone**: `ZONE="us-central1-a"`
- **Machine Type**: `MACHINE_TYPE="e2-standard-4"`
- **Disk Size**: `DISK_SIZE="200GB"`

### Rate Limiting

To adjust scraping rate, edit `shard_runner.py`:
- `max_concurrent=50` - Concurrent workers
- `rate_limit=100` - Requests per second

## Troubleshooting

### Instance Not Starting

1. **Check startup script logs:**
   ```bash
   gcloud compute instances get-serial-port-output grokipedia-shard-0
   ```

2. **Verify credentials in `.env`:**
   - All required variables present?
   - GitHub PAT has repo access?
   - BrightData credentials valid?

### Scraper Not Running

1. **SSH into instance:**
   ```bash
   gcloud compute ssh grokipedia-shard-0
   ```

2. **Check logs:**
   ```bash
   tail -f /var/log/grokipedia-scraper.log
   ```

3. **Verify virtual environment:**
   ```bash
   cd /home/wiki-grok-comparison
   source venv/bin/activate
   python scripts/shard_runner.py --start_idx 0 --end_idx 10
   ```

### GCS Upload Failures

1. **Check service account permissions:**
   ```bash
   # On the VM
   curl "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email" -H "Metadata-Flavor: Google"
   ```

2. **Verify bucket exists:**
   ```bash
   gsutil ls gs://enwiki-structured-contents-20251028/
   ```

3. **Check quota/limits:**
   - GCS API quota not exceeded
   - Disk space on VM sufficient

### URLs Not Loading

1. **HuggingFace dataset:**
   - Verify `HF_API_TOKEN` is set
   - Check dataset is accessible

2. **Local file:**
   - Ensure file was uploaded to GCS
   - Check path in instance metadata

## Cost Considerations

- **VM Costs**: ~$0.17/hour for `e2-standard-4` in us-central1-a
- **GCS Storage**: ~$0.023/GB/month (standard)
- **Network Egress**: Free within same region

**Example**: 10 shards running for 24 hours ≈ $40.80 in VM costs

## Notes

- Instances use the default Compute Engine service account for GCS access (no key file needed)
- Each shard processes its range independently, so partial completion is safe
- Results are saved incrementally to GCS (every 1000 items)
- Failed URLs are logged but don't stop the scraping process
- Instance names must be unique (shard ID ensures this)

