#!/bin/bash
# setup-loadgen-netns.sh — configure the secondary ENI into the "loadgen"
# network namespace for dual-ENI real-NIC benchmarking.
#
# Usage: sudo ./setup-loadgen-netns.sh <ENI_MAC> <ENI_PRIVATE_IP/PREFIX> <GATEWAY>
# Example: sudo ./setup-loadgen-netns.sh 0e:5c:94:9d:0e:73 172.31.45.237/20 172.31.32.1
#
# Idempotent: safe to re-run (e.g. after reboot — netns does NOT persist).
# NEVER touches the primary interface. Finds the ENI by MAC address, so it
# works regardless of the kernel's interface naming on each host.
set -euo pipefail
MAC=${1:?ENI MAC required}
IPCIDR=${2:?ENI private IP/prefix required}
GW=${3:?gateway required}
NS=loadgen

# Locate iface by MAC — check default ns, then the netns (already-done case).
IFACE=$(ip -o link | awk -v m="$MAC" 'tolower($0) ~ tolower(m) {gsub(":","",$2); print $2}' | head -1)
if [ -z "$IFACE" ]; then
  if ip netns exec $NS ip -o link 2>/dev/null | grep -qi "$MAC"; then
    echo "already configured: $MAC lives in netns $NS"; ip netns exec $NS ip -br addr; exit 0
  fi
  echo "ERROR: no interface with MAC $MAC found"; exit 1
fi

ip netns add $NS 2>/dev/null || true
ip link set "$IFACE" netns $NS
ip netns exec $NS ip link set lo up
ip netns exec $NS ip addr add "$IPCIDR" dev "$IFACE" 2>/dev/null || true
ip netns exec $NS ip link set "$IFACE" up
ip netns exec $NS ip route replace default via "$GW" dev "$IFACE"
# remove any policy-routing rules ec2-net-utils created for this ENI in the default ns
IP=${IPCIDR%/*}
while ip rule show | grep -q "from $IP"; do ip rule del from "$IP" || break; done
echo "netns $NS ready:"; ip netns exec $NS ip -br addr | grep -v '^lo'
