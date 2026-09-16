# ───────────────────────── reconstruction pipeline paths ─────────────────────────
# Sourced by run_pipeline.sh; the Python scripts read these via os.environ

# Root of this extracted project 
RECON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RECON_ROOT



export SCRIPTS_DIR="$RECON_ROOT/scripts"




export ENV_SAM3=sam3
export ENV_SAM3D=sam3d
export ENV_HAWOR=hawor
export ENV_TAPNET=tapnet
export ENV_MOGE=moge
export ENV_RECGEN=recgen
export ENV_WUJI=wuji_retargeting
export ENV_GENESIS=genesis

# ── External checkouts ──
export RECGEN_DIR="${RECGEN_DIR:-$HOME/recgen}"
export MOGE_REPO="${MOGE_REPO:-$HOME/MoGe}"
export HAWOR_DIR="${HAWOR_DIR:-$HOME/do-as-i-do/reconstruction/modules/HaWoR}"
export WUJI_RETARGETING_DIR="${WUJI_RETARGETING_DIR:-$HOME/wuji-ego-mint/eval/simulate/wuji-retargeting}"
export TAPNET_CKPT="${TAPNET_CKPT:-$HOME/do-as-i-do/reconstruction/weights/tapnet/bootstapir_checkpoint_v2.pt}"

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
