# ─────────────────────────────────────────────────────────────
# KTC-Vis Dockerfile
# Base: dolfinx official image (provides FEniCS/dolfinx for CUQI8)
# ─────────────────────────────────────────────────────────────
FROM dolfinx/dolfinx:v0.7.2

LABEL maintainer="KTC-Vis Team"
LABEL description="KTC-Vis: Interactive EIT Algorithm Benchmarking Dashboard"

# Install Miniconda inside the FEniCS image
ENV CONDA_DIR=/opt/conda
RUN apt-get update && apt-get install -y wget curl && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    apt-get clean

ENV PATH=$CONDA_DIR/bin:$PATH

# Copy and install Conda environment
WORKDIR /app
COPY environment.yml .
# Skip fenics-dolfinx in conda (already installed in base image)
RUN conda env create -f environment.yml --name ktc-vis || true
SHELL ["conda", "run", "-n", "ktc-vis", "/bin/bash", "-c"]

# Copy project source
COPY . /app/

# Expose Dash default port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# Entry point
CMD ["conda", "run", "--no-capture-output", "-n", "ktc-vis", "python", "app.py"]
