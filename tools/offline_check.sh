#!/bin/sh
# offline_check.sh - run any command with the network blocked and every connection attempt logged.
#
#   tools/offline_check.sh python voice_match.py read.m4a ref.mp3 -o out.wav --best-of
#
# Blocks the network for Python processes (tools/netguard/sitecustomize.py) and points HTTP(S) proxies at a
# closed port, so anything that slips past the guard fails too. Prints the number of attempts at the end;
# exits non-zero if there were any.
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="${NETGUARD_LOG:-$(mktemp -t netguard)}"
: > "$LOG"
NETGUARD_LOG="$LOG" PYTHONPATH="$HERE/netguard${PYTHONPATH:+:$PYTHONPATH}" \
  HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 \
  "$@"
status=$?
attempts=$(grep -c '^\[pid' "$LOG")
echo "netguard: $attempts network attempt(s) (log: $LOG)"
[ "$attempts" -eq 0 ] && exit $status
head -40 "$LOG"
exit 3
