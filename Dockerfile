FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
 apt-get install -y --no-install-recommends \
 python3 python3-dev python3-venv python3-distutils \
 build-essential git curl ca-certificates cmake

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
 mv ~/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace
# COPY pyproject.toml .
COPY pyproject.toml uv.lock ./

ENV UV_LINK_MODE=copy
# RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -r pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

COPY . .
# RUN uv pip install --system --no-deps .
RUN uv sync --frozen

ENV PYTHONUNBUFFERED=1
CMD ["uv", "run", "python3", "dev/main.py", "-m", "loss=ce,ce_before_avg_us", "reg_loss_fn=ce,bce"]