#!/usr/bin/env bash
# Create the conda envs the reconstruction pipelines switch between.
#
#   ./envs/create_envs.sh                 # every env below that does not exist yet
#   ./envs/create_envs.sh genesis sam3    # just these
#
# Env names match config/paths.sh (ENV_*). Existing envs are left alone; remove one
# with `conda env remove -n NAME` to rebuild it. Each yml's header lists the packages
# that were installed from local checkouts and must be installed by hand afterwards.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -gt 0 ]; then
    ENVS=("$@")
else
    ENVS=()
    for yml in "$HERE"/*.yml; do ENVS+=("$(basename "$yml" .yml)"); done
fi

for name in "${ENVS[@]}"; do
    yml="$HERE/$name.yml"
    [ -f "$yml" ] || { echo "no $yml" >&2; exit 1; }
    if conda env list | awk '{print $1}' | grep -qx "$name"; then
        echo "=== $name: exists, skipping ==="
        continue
    fi
    echo "=== $name: creating from $(basename "$yml") ==="
    conda env create -f "$yml"
    if grep -q "^#   pip install" "$yml"; then
        echo "  !! $name needs local installs by hand:"
        grep "^#   pip install" "$yml" | sed 's/^#  /    /'
    fi
done
