#!/usr/bin/env bash
# Aggregator: pulls status.json from all benchmark hosts and commits to dashboard repo.
# Run via cron on benchdev every 2 minutes.
#
# Usage: ./aggregate-status.sh
# Cron:  */2 * * * * /home/rainval/valkey-benchmark-dashboard/aggregate-status.sh

set -euo pipefail

DASHBOARD_DIR="$HOME/valkey-benchmark-dashboard"
STATUS_DIR="$DASHBOARD_DIR/data/status"
SSH_KEY="$HOME/.ssh/openssh-ec2-pair.pem"
SSH_OPTS="-F /dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i $SSH_KEY"

# Benchmark hosts
HOSTS=(
    "ec2-user@arm.conductress.rainsupreme.net"
    "ec2-user@x86.conductress.rainsupreme.net"
    "ec2-user@intel.conductress.rainsupreme.net"
)

mkdir -p "$STATUS_DIR"

# Pull status from each host (continue on failure)
for host in "${HOSTS[@]}"; do
    hostname=$(echo "$host" | cut -d@ -f2 | cut -d. -f1)
    # shellcheck disable=SC2086
    if ssh $SSH_OPTS "$host" "cat ~/conductress/status/status.json" > "$STATUS_DIR/${hostname}.json" 2>/dev/null; then
        :
    else
        # Write a minimal offline status
        echo "{\"host\": \"${hostname}\", \"runner\": {\"state\": \"unreachable\"}, \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$STATUS_DIR/${hostname}.json"
    fi
done

# Commit and push if changed
cd "$DASHBOARD_DIR"
if ! git diff --quiet data/status/ 2>/dev/null; then
    git add data/status/
    git commit -m "status: update $(date -u +%H:%M)" --no-gpg-sign -q
    git push -q
fi
