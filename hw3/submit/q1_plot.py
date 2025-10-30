import os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Read logs from the submission directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "submit", "data")

def _runs(prefix: str):
    return sorted(glob.glob(os.path.join(DATA_ROOT, f"{prefix}*")))


def _load_one(run_dir: str, tag_order=("Eval_AverageReturn", "Train_AverageReturn", "AverageReturn")):
    ev = EventAccumulator(run_dir); ev.Reload()
    for tag in tag_order:
        try:
            scalars = ev.Scalars(tag)
        except KeyError:
            scalars = None
        if scalars:
            steps = np.array([s.step for s in scalars], dtype=np.float64)
            vals  = np.array([s.value for s in scalars], dtype=np.float32)
            return steps, vals, tag
    return None, None, None


def load_group(prefix: str):
    """Return (steps, values_matrix, used_tag).
    values_matrix has shape [num_seeds, num_steps] and is interpolated
    to the step grid from the first usable run so we can take mean/std.
    """
    steps0 = None; tag_used = None; rows = []
    for i, d in enumerate(_runs(prefix)):
        s, v, tag = _load_one(d)
        if s is None:  # skip runs without the expected scalar
            continue
        if steps0 is None:
            steps0, tag_used = s, tag
        # align each seed to the common grid by interpolation
        order = np.argsort(s); s, v = s[order], v[order]
        rows.append(np.interp(steps0, s, v))
    if not rows:
        raise FileNotFoundError(f"No usable runs for '{prefix}' under {DATA_ROOT}")
    return steps0, np.stack(rows, 0), tag_used

dqn_steps,  dqn_mat,  dqn_tag  = load_group("q1_dqn_")
ddqn_steps, ddqn_mat, ddqn_tag = load_group("q1_doubledqn_")

if not np.array_equal(dqn_steps, ddqn_steps):
    lo = max(dqn_steps.min(), ddqn_steps.min())
    hi = min(dqn_steps.max(), ddqn_steps.max())
    grid = np.linspace(lo, hi, num=min(len(dqn_steps), len(ddqn_steps)))
    dqn_mat  = np.vstack([np.interp(grid, dqn_steps,  r) for r in dqn_mat])
    ddqn_mat = np.vstack([np.interp(grid, ddqn_steps, r) for r in ddqn_mat])
    dqn_steps = ddqn_steps = grid

# mean ± std across seeds
m_dqn,  s_dqn  = dqn_mat.mean(0),  dqn_mat.std(0)
m_ddqn, s_ddqn = ddqn_mat.mean(0), ddqn_mat.std(0)

plt.figure(figsize=(8, 6))
plt.plot(dqn_steps,  m_dqn,  label="DQN",  linewidth=2.0)
plt.fill_between(dqn_steps,  m_dqn - s_dqn,  m_dqn + s_dqn,  alpha=0.18)
plt.plot(ddqn_steps, m_ddqn, label="DDQN", linewidth=2.0)
plt.fill_between(ddqn_steps, m_ddqn - s_ddqn, m_ddqn + s_ddqn, alpha=0.18)

plt.title("DQN vs DDQN on LunarLander-v3", fontsize=12)
plt.xlabel("Training Steps")
plt.ylabel("Average per-epoch reward")
plt.ticklabel_format(style="sci", axis="x", scilimits=(5, 5))
plt.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig("q1_dqn_vs_ddqn.png", dpi=240)
print("Saved q1_dqn_vs_ddqn.png")