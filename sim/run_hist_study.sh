#!/usr/bin/env bash
# Train the history policy on L4 data, then evaluate it and the no-history control at several exec horizons.
# Put in ~/humanoid_ws/sim and run: bash run_hist_study.sh   (resumable: skips finished steps)
set -e
cd ~/humanoid_ws/sim
source /opt/ros/jazzy/setup.bash
source ../venv/bin/activate
mkdir -p ../results ../ckpt

# 1. train (~13 min)
if [ ! -f ../ckpt/bc_L4_hist.pt ]; then
  python train_bc.py --data ../data_L4 --out ../ckpt/bc_L4_hist.pt --hist 5 10 15 --steps 15000 \
    | tee ../results/train_L4_hist.log
fi

# 2. determinism check: same seed twice, outcomes should match
for R in a b; do
  [ -f ../results/det_$R.json ] || python eval_policy.py --ckpt ../ckpt/bc_L4_hist.pt --level L4 \
    --episodes 2 --exec 5 --json ../results/det_$R.json > /dev/null
done
python - <<'EOF'
import json
a, b = (json.load(open(f"../results/det_{r}.json"))["results"] for r in "ab")
for x, y in zip(a, b):
    print("determinism seed", x["seed"], "depth", round(x["depth"], 4), round(y["depth"], 4),
          "OK" if abs(x["depth"] - y["depth"]) < 1e-6 else "DIFFERENT")
EOF

# 3. history policy at several horizons, then the no-history control on the same seeds
run() {  # name ckpt exec
  [ -f ../results/$1.json ] || python eval_policy.py --ckpt $2 --level L4 --episodes 30 --exec $3 \
    --json ../results/$1.json > ../results/$1.log
}
for E in 5 2 1 10; do run L4_hist_exec$E ../ckpt/bc_L4_hist.pt $E; done
for E in 1 2 5; do run L4_nohist_exec$E ../ckpt/bc_L4.pt $E; done

python summarize.py ../results/L4_*exec*.json
