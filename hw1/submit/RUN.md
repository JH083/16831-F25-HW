# RUN.md — How to reproduce results

This README lists the exact commands and hyperparameters used to produce the
numbers/plots in the report, and the run logs included under `run_logs/`.

---

## 0) Environment

- **Python**: 3.8  
- **Key packages** (versions used to run the HW):
  - `gym==0.25.1`
  - `mujoco-py==2.1.2.14`
  - `numpy==1.24.4`
  - `Cython<3.0`
  - `torch==1.12.1+cpu`, `torchvision==0.13.1+cpu`
  - `tensorboard==2.x` (for events)
- **Headless GL**: `MUJOCO_GL=osmesa`

> **Note**: Videos are disabled for all submitted runs (`--video_log_freq -1`)
to keep file sizes small, as requested.

---

## 1) One-time shell setup (from repo root that contains `rob831/`)

```bash
export PYTHONPATH=.
export MUJOCO_GL=osmesa
unset LD_PRELOAD
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu
```
---

## 2) Behavioral Cloning (Q1 / Part 2)

### Ant-v2 (used in Table 2 / Figure 1 baseline)

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --env_name Ant-v2 \
  --exp_name bc_ant \
  --n_iter 1 \
  --n_layers 5 \
  --learning_rate 4e-3 \
  --eval_batch_size 5000 \
  --video_log_freq -1 \
  --no_gpu
```


### Humanoid-v2 (used in Table 2 / Figure 2 baseline)

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Humanoid.pkl \
  --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
  --env_name Humanoid-v2 \
  --exp_name bc_humanoid \
  --n_iter 1 \
  --n_layers 5 \
  --learning_rate 4e-3 \
  --eval_batch_size 5000 \
  --video_log_freq -1 \
  --no_gpu
```

Eval setting: episode length 1000, --eval_batch_size 5000 ⇒ ~5 rollouts per evaluation.
The logged Eval_AverageReturn/Eval_StdReturn are mean ± std across those rollouts.

---

## 3) Q1 / Part 4 — Hyperparameter sweep (Ant-v2)

Hyperparameter: number of BC training steps per iteration (--num_agent_train_steps_per_iter).
I ran: 250, 500, 1000, 2000, 3000, 4000.

```bash
for steps in 250 500 1000 2000 3000 4000; do
  python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --env_name Ant-v2 \
    --exp_name p4_ant_steps_${steps} \
    --n_iter 1 \
    --num_agent_train_steps_per_iter ${steps} \
    --n_layers 5 \
    --learning_rate 4e-3 \
    --eval_batch_size 5000 \
    --video_log_freq -1 \
    --no_gpu
done
```

(I then parsed the runs to make Figure 1: performance vs training steps.)


---

## 4) DAgger (Q2 / Part 2)

### Ant-v2

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --env_name Ant-v2 \
  --exp_name dagger_ant \
  --n_iter 8 \
  --do_dagger \
  --n_layers 3 \
  --learning_rate 4e-3 \
  --eval_batch_size 5000 \
  --video_log_freq -1 \
  --no_gpu
```

### Humanoid-v2

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Humanoid.pkl \
  --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
  --env_name Humanoid-v2 \
  --exp_name dagger_humanoid \
  --n_iter 8 \
  --do_dagger \
  --n_layers 3 \
  --learning_rate 4e-3 \
  --eval_batch_size 5000 \
  --video_log_freq -1 \
  --no_gpu
```
