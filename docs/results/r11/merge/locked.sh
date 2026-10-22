#!/bin/bash
# locked.sh TAG CMD...: runs CMD under the heavy lock
LOCK=C:/Users/User/Desktop/docunet_bench/.heavy_lock
TAG=$1; shift
until mkdir $LOCK 2>/dev/null; do
  if [ -f $LOCK/owner ] && [ $(( $(date +%s) - $(stat -c %Y $LOCK/owner) )) -gt 7200 ]; then echo "stale lock: $(cat $LOCK/owner)"; rm -rf $LOCK; continue; fi
  sleep 20
done
echo "merge $TAG $(date)" > $LOCK/owner
trap 'rm -rf C:/Users/User/Desktop/docunet_bench/.heavy_lock' EXIT
echo "lock taken $(date +%T)"
"$@"
echo "exit $? $(date +%T)"
