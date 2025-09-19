# RUN.md — how to reproduce the submitted results

## Environment (tested)
- Python 3.8
- gym==0.21.0
- numpy==1.23.5
- mujoco-py==2.1.2.14
- glfw==2.5.5, PyOpenGL==3.1.7, Cython==0.29.36
- Headless rendering: `MUJOCO_GL=osmesa`

**Before running**, from the repo root (the folder that contains `rob831/`):
```bash
cd /content/16831-F25-HW/hw1
export PYTHONPATH=.
export MUJOCO_GL=osmesa
unset LD_PRELOAD
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu
