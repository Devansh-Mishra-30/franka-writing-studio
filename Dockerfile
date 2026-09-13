FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Install operating-system dependencies only.
# Python packages will be isolated inside /opt/venv.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglu1-mesa \
        libglew2.2 \
        libglib2.0-0 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated Python environment.
# Runtime libraries required by Qt / PySide6 / pyqtgraph.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libxkbcommon-x11-0 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-randr0 \
        libxcb-shape0 \
        libxcb-xfixes0 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

# Make the virtual environment's Python and pip the default.
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade packaging tools before installing project dependencies.
RUN python -m pip install --no-cache-dir --upgrade \
        pip \
        setuptools \
        wheel

# Copy only the dependency list first.
# This allows Docker to cache dependency installation separately from source code.
COPY requirements.lock /tmp/requirements.lock

# Install all Python dependencies inside the virtual environment.
RUN python -m pip install --no-cache-dir \
        -r /tmp/requirements.lock

# Fail the image build immediately if a required dependency cannot import.
RUN python - <<'PY'
import importlib

modules = [
    "numpy",
    "scipy",
    "matplotlib",
    "pybullet",
    "pybullet_data",
    "pinocchio",
    "imageio",
    "imageio_ffmpeg",
    "PySide6",
    "pyqtgraph",
    "pytest",
]

for module_name in modules:
    importlib.import_module(module_name)
    print(f"[PASS] imported {module_name}")
PY

WORKDIR /workspace

CMD ["bash"]
