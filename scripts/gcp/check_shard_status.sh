#!/bin/bash
# Helper script to check status of fixed issues shard instances

SHARD_ID=${1:-0}
ZONE="us-central1-a"
INSTANCE_NAME="fixed-issues-shard-${SHARD_ID}"

echo "Checking status of ${INSTANCE_NAME}..."
echo ""

# Check if instance exists
if ! gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} &>/dev/null; then
    echo "❌ Instance ${INSTANCE_NAME} does not exist"
    exit 1
fi

# Check instance status
STATUS=$(gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --format="value(status)")
echo "Instance status: ${STATUS}"

if [ "$STATUS" != "RUNNING" ]; then
    echo "⚠️  Instance is not running. Check with:"
    echo "   gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE}"
    exit 1
fi

echo ""
echo "1. Checking startup script logs (serial console):"
echo "   gcloud compute instances get-serial-port-output ${INSTANCE_NAME} --zone=${ZONE} | tail -50"
echo ""

echo "2. Checking if scraper log exists:"
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command "ls -lh /var/log/fixed-issues-scraper.log 2>/dev/null || echo 'Log file does not exist yet'"
echo ""

echo "3. Checking if Python process is running:"
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command "ps aux | grep fixed_issues_runner || echo 'Process not found'"
echo ""

echo "4. Checking if repository exists:"
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command "ls -d /home/wiki-grok-comparison 2>/dev/null || echo 'Repository not found'"
echo ""

echo "5. Checking recent system logs:"
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command "sudo journalctl -u google-startup-scripts.service --no-pager -n 20 2>/dev/null || echo 'Cannot access system logs'"
echo ""

echo "To view live logs (once they exist):"
echo "   gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command 'tail -f /var/log/fixed-issues-scraper.log'"
echo ""
echo "To SSH into the instance:"
echo "   gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"

