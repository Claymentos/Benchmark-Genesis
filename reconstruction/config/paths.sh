# ───────────────────────── reconstruction pipeline paths ─────────────────────────
# Sourced by every run_*.sh pipeline; the Python scripts read these via os.environ.
# Each ${VAR:-default} can be overridden from the environment.

# Root of the reconstruction/ directory
RECON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RECON_ROOT
export SCRIPTS_DIR="$RECON_ROOT/scripts"

# ── Conda env per stage (see envs/) ──
export ENV_SAM3=sam3
export ENV_SAM3D=sam3d
export ENV_HAWOR=hawor
export ENV_TAPNET=tapnet
export ENV_MOGE=moge
export ENV_RECGEN=recgen
export ENV_WUJI=wuji_retargeting
export ENV_GENESIS=genesis
export ENV_FOUNDATIONPOSE=foundationpose
# SuperDex replaces Genesis as the simulator. Conda rather than the uv venv the project
# documents, to match every other stage here; superdex pins CPython 3.12.
export ENV_SUPERDEX=superdex
# Lightning Grasp's own env: its CUDA kernels ship as binaries built against a pinned
# python/torch pair, so it cannot share one of the others.
export ENV_LYGRA=lygra
# smplx + CUDA torch; MANO fitting (mano_from_mocap.py) and EC-Fit.
export ENV_ECFIT=ecfit
# The ZED SDK's pyzed is installed against the conda base interpreter, not into any
# of the per-stage envs, so this one stage steps outside the env chain.
export PYTHON_ZED="${PYTHON_ZED:-$(conda info --base)/bin/python}"
export ZED_DEPTH_MODE="${ZED_DEPTH_MODE:-NEURAL}"

# ── External checkouts ──
export RECGEN_DIR="${RECGEN_DIR:-$HOME/recgen}"
export MOGE_REPO="${MOGE_REPO:-$HOME/MoGe}"
export HAWOR_DIR="${HAWOR_DIR:-$HOME/do-as-i-do/reconstruction/modules/HaWoR}"
export WUJI_RETARGETING_DIR="${WUJI_RETARGETING_DIR:-$HOME/wuji-ego-mint/eval/simulate/wuji-retargeting}"
export TAPNET_CKPT="${TAPNET_CKPT:-$HOME/do-as-i-do/reconstruction/weights/tapnet/bootstapir_checkpoint_v2.pt}"
export FOUNDATIONPOSE_DIR="${FOUNDATIONPOSE_DIR:-$HOME/FoundationPose}"
# Vendored in-tree, because the Wuji config in scripts/wuji_lygra.py registers against
# this checkout's robot factory.
export LIGHTNING_GRASP_DIR="${LIGHTNING_GRASP_DIR:-$RECON_ROOT/lightning-grasp}"
# Panda with the Wuji Hand already mounted on the flange, one URDF per side. Its
# mount transform is calibrated, unlike one built by joining the two source URDFs
# (scripts/make_franka_wuji_urdf.py, for other arms or hands). Set PANDA_WUJI_URDF to
# override the whole path instead of composing it from the side.
export PANDA_WUJI_URDF_DIR="${PANDA_WUJI_URDF_DIR:-$HOME/genesis-world/robots/panda_wuji/urdf}"

# ── Models ──
export RECGEN_CHECKPOINT="${RECGEN_CHECKPOINT:-recgen_base.multiview_stereo}"
export MOGE_CHECKPOINT="${MOGE_CHECKPOINT:-Ruicheng/moge-3-vitl}"

# ── Host / GPU ──
export CUDA_VISIBLE_DEVICES=0
# X display used by the click-based SAM3 segmentation UI (Stage 1).
export SAM3_DISPLAY=:1
# TAPIR over a whole sequence at once can exhaust VRAM when another job shares the
# GPU; the tracker chunks, and this keeps allocator fragmentation from compounding.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
