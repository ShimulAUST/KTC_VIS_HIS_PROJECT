#!/usr/bin/env bash
# Pulls the three KTC2023 algorithm Docker images from Docker Hub.
# Run once after cloning KTC-Vis:
#
#   bash scripts/setup_third_party.sh
#
# Re-running is safe — Docker skips images that are already up to date.
#
# Algorithm run commands (for reference):
#
#   ABC1:
#     docker run --rm -v "${PWD}:/app" muzammal5566/ktc2023-abc-python:latest \
#       python main_python.py TrainingData Outputs <level>
#
#   CUQI8:
#     docker run --rm -v "${PWD}/Output/difficulty_<level>:/app/Output" \
#       muzammal5566/ktc2023-cuqi8:latest \
#       conda run --no-capture-output -n env python main.py TrainingData Output <level>
#
#   PNPE2E:
#     docker run --rm -v "${PWD}:/app" muzammal5566/ktc2023-pnpe2e:latest \
#       python -c "from main import main; main('/app/TrainingData', '/app/Output_difficulty_<level>', <level>)"

set -euo pipefail

ABC1_IMAGE="muzammal5566/ktc2023-abc-python:latest"
CUQI8_IMAGE="muzammal5566/ktc2023-cuqi8:latest"
PNPE2E_IMAGE="muzammal5566/ktc2023-pnpe2e:latest"

# Images are built for linux/amd64. On Apple Silicon (M1/M2/M3) Macs
# Docker Desktop runs them via Rosetta — the --platform flag is required.
PLATFORM="linux/amd64"

pull_image() {
    local image="$1"
    echo ""
    echo "[PULL] $image (platform: $PLATFORM) ..."
    docker pull --platform "$PLATFORM" "$image"
    echo "[OK]   $image ready."
}

# Check Docker is available
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not on PATH. Install Docker Desktop first."
    exit 1
fi

pull_image "$ABC1_IMAGE"
pull_image "$CUQI8_IMAGE"
pull_image "$PNPE2E_IMAGE"

echo ""
echo "All algorithm images pulled successfully."
echo ""
docker images | grep -E "REPOSITORY|ktc2023"
