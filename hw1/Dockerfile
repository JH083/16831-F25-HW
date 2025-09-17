# Force an x86_64 (amd64) userspace so mujoco-py builds reliably
FROM python:3.10-bullseye

# System packages needed by mujoco-py and headless rendering
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gcc g++ make cmake \
    libgl1-mesa-glx libosmesa6-dev libglfw3 libglew-dev \
    patchelf ffmpeg \
    git wget unzip \
    xvfb \
 && rm -rf /var/lib/apt/lists/*

# Create a workspace
WORKDIR /workspace

# (Optional) Speed up pip & silence cache
ENV PIP_NO_CACHE_DIR=1

# Pin Python deps compatible with old Gym + mujoco-py used in HW1
# Adjust if your requirements.txt already pins these
RUN python -m pip install --upgrade pip && \
    pip install "numpy<2.0" "gym==0.25.1" "mujoco-py>=2.1,<2.2" Cython \
               tensorboard==2.10.0 tensorboardX==2.5.1 \
               "matplotlib>=3.8,<3.10" \
               torch==1.12.1

# MuJoCo-py env tweaks for headless CPUs
ENV MUJOCO_PY_SKIP_LOWER_GL_VERSION_CHECK=1
ENV MUJOCO_GL=osmesa

# By default we’ll mount your repo at /workspace; nothing else needed here