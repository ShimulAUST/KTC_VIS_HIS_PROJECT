# ─────────────────────────────────────────────────────────────────
# KTC-Vis Development & Production Image
#
# Build:  docker build -t ktc-vis .
# Run:    docker compose up
#
# The algorithm containers (ABC1, CUQI8, PNPE2E) are separate images
# invoked via the host Docker socket — they do NOT live inside this image.
# ─────────────────────────────────────────────────────────────────

FROM continuumio/miniconda3:23.10.0-1

LABEL maintainer="KTC-Vis Team"
LABEL description="KTC-Vis: Interactive EIT Algorithm Benchmarking Dashboard"

# Install Docker CLI (static binary — works on any Debian version)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz \
        | tar xz --strip-components=1 -C /usr/local/bin docker/docker

WORKDIR /app

# ── Install Conda environment ─────────────────────────────────────
COPY environment.yml .
RUN conda env create -f environment.yml --name ktc-vis && \
    conda clean -afy

# Make conda env the default shell for all subsequent RUN/CMD
SHELL ["conda", "run", "-n", "ktc-vis", "/bin/bash", "-c"]

# ── Install project in editable mode ─────────────────────────────
COPY pyproject.toml ./
COPY ktc_vis/ ./ktc_vis/
RUN pip install -e . --no-deps

# ── Copy remaining source ─────────────────────────────────────────
COPY app.py ./
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# ── Runtime config ────────────────────────────────────────────────
EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

CMD ["conda", "run", "--no-capture-output", "-n", "ktc-vis", "python", "app.py"]
