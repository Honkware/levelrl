#!/usr/bin/env bash
# Builds lc0 (CPU, Eigen backend) for the Maia-1 cross-check and queues the cross-check to run after the main
# pipeline finishes, since extra CPU load during training would weaken the fixed-movetime rungs.
set -e
cd ~/lc0
rm -rf build/release
# lc0 0.31's BLAS backend falls back to its bundled Eigen when no OpenBLAS or MKL is installed, which is the case here.
PATH=~/levelrl/.venv/bin:$PATH ./build.sh -Dgtest=false -Dcudnn=false -Dplain_cuda=false -Dopencl=false -Dispc=false
ln -sf ~/lc0/build/release/lc0 ~/lc0/build/lc0
~/lc0/build/lc0 --help | head -2
tmux has-session -t crosscheck 2>/dev/null && { echo "cross-check already queued"; exit 0; }
tmux new -d -s crosscheck 'cd ~/levelrl; while tmux has-session -t pipeline 2>/dev/null; do sleep 300; done; rm -f runs/pipeline/crosscheck.json runs/pipeline/report.json; scripts/run_pipeline.sh'
echo "lc0 built; the cross-check will run once the pipeline finishes"
