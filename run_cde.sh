#!/bin/bash
# run_cde.sh
# Run 2 py files

echo "Starting job at $(date)" > job.log

python3 run.py --dataset Libras --linsvc >> job.log 2>&1
echo "Linsvc finished at $(date)" >> job.log

python3 run.py --dataset Libras >> job.log 2>&1
echo "Kernel finished at $(date)" >> job.log