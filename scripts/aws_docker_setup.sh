#!/bin/bash
set -e
SYSTEM_MEMORY_MB_75_PERCENT=$(free|head -2|tail -1|awk '{ print $2*.75M }')
mount -o remount,size=$SYSTEM_MEMORY_MB_75_PERCENT /dev/shm
mkdir -p /scratch/tmp
export TMP=/scratch/tmp
export TMP_DIR=/scratch/tmp
export TMPDIR=/scratch/tmp
export TEMP=/scratch/tmp
#date
#echo "**** Command ****"
#echo "$@"
#echo "**** Filesystem ****"
#df -h
#echo "**** Environment ****"
#env
#echo "jobId: $AWS_BATCH_JOB_ID"
#echo "jobQueue: $AWS_BATCH_JQ_NAME"
#echo "computeEnvironment: $AWS_BATCH_CE_NAME"
#echo "**** AWS INFO ****"
#echo "Public-ip:"
## add || true so that this works locally as well
#curl --connect-timeout .1 -s http://169.254.169.254/latest/meta-data/public-ipv4 || true
#echo ""
#echo "Instance type:"
## add || true so that this works locally as well
#curl --connect-timeout .1 -s http://169.254.169.254/latest/meta-data/instance-type || true
#echo ""
#echo "Instance id:"
## add || true so that this works locally as well
#curl --connect-timeout .1 -s http://169.254.169.254/latest/meta-data/instance-id || true
#echo ""
#echo "**** Running Command ****"
exec "$@"
echo "**** Command Finished ****"
date