#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

TORCH_TARGET="${CHEETAH_TORCH_TARGET:-auto}"
CUDA_VERSION="${CHEETAH_CUDA_VERSION:-auto}"
PYTHON_OVERRIDE="${PYTHON:-}"
VENV_DIR="${CHEETAH_VENV:-.venv}"
USE_VENV=1
INSTALL_DEV=0
DRY_RUN=0

SUPPORTED_CUDA_TAGS="cu118 cu126 cu128 cu130"

print_banner() {
    local reset bold
    local colors

    # Logo text mirrors cheetah/tui/main_menu.py.
    if [ -n "${NO_COLOR:-}" ]; then
        reset=""
        bold=""
        colors=("" "" "" "" "")
    else
        reset=$'\033[0m'
        bold=$'\033[1m'
        colors=(
            $'\033[38;5;51m'
            $'\033[38;5;45m'
            $'\033[38;5;83m'
            $'\033[38;5;226m'
            $'\033[38;5;201m'
        )
    fi

    printf '\n'
    printf '%s%s%s\n' "${colors[0]}" '░░      ░░░  ░░░░  ░░        ░░        ░░        ░░░      ░░░  ░░░░  ░' "$reset"
    printf '%s%s%s\n' "${colors[1]}" '▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒' "$reset"
    printf '%s%s%s\n' "${colors[2]}" '▓  ▓▓▓▓▓▓▓▓        ▓▓      ▓▓▓▓      ▓▓▓▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓  ▓▓        ▓' "$reset"
    printf '%s%s%s\n' "${colors[3]}" '█  ████  ██  ████  ██  ████████  ███████████  █████        ██  ████  █' "$reset"
    printf '%s%s%s\n' "${colors[4]}" '██      ███  ████  ██        ██        █████  █████  ████  ██  ████  █' "$reset"
    printf '%s%s%s\n\n' "$bold" 'Cheetah installer' "$reset"
}

usage() {
    cat <<'EOF'
Usage: ./setup.sh [options]

Creates a Python virtual environment, installs the correct PyTorch wheel, then
installs Cheetah in editable mode.

Options:
  --torch TARGET       auto, cpu, mps, cuda, cu118, cu126, cu128, or cu130
  --cuda VERSION      CUDA version to use for selection, e.g. 12.8 or 13.0
  --cpu               Shortcut for --torch cpu
  --mps               Shortcut for --torch mps (macOS PyTorch default wheel)
  --dev               Install Cheetah with its dev extras
  --python PATH       Python executable to use before the venv is created
  --venv PATH         Virtual environment directory (default: .venv)
  --no-venv           Install into the current Python environment
  --dry-run           Print the commands that would run
  -h, --help          Show this help

Environment overrides:
  CHEETAH_TORCH_TARGET=auto|cpu|mps|cuda|cu118|cu126|cu128|cu130
  CHEETAH_CUDA_VERSION=12.8
  CHEETAH_VENV=.venv
  PYTHON=python3.12

Android notes:
  Android/Termux is CPU-only here. The script first tries an existing torch
  install, then a Termux python-pytorch package if available, then pip CPU
  wheels. Official PyTorch pip wheels may not exist for every Android setup.
EOF
}

log() {
    printf '\033[1;34m==>\033[0m %s\n' "$*" >&2
}

warn() {
    printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2
}

die() {
    printf '\033[1;31merror:\033[0m %s\n' "$*" >&2
    exit 1
}

run() {
    log "+ $*"
    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi
    "$@"
}

lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --torch)
            [ "$#" -ge 2 ] || die "--torch requires a value"
            TORCH_TARGET="$2"
            shift 2
            ;;
        --torch=*)
            TORCH_TARGET="${1#*=}"
            shift
            ;;
        --cuda)
            [ "$#" -ge 2 ] || die "--cuda requires a value"
            CUDA_VERSION="$2"
            shift 2
            ;;
        --cuda=*)
            CUDA_VERSION="${1#*=}"
            shift
            ;;
        --cpu)
            TORCH_TARGET="cpu"
            shift
            ;;
        --mps)
            TORCH_TARGET="mps"
            shift
            ;;
        --dev)
            INSTALL_DEV=1
            shift
            ;;
        --python)
            [ "$#" -ge 2 ] || die "--python requires a value"
            PYTHON_OVERRIDE="$2"
            shift 2
            ;;
        --python=*)
            PYTHON_OVERRIDE="${1#*=}"
            shift
            ;;
        --venv)
            [ "$#" -ge 2 ] || die "--venv requires a value"
            VENV_DIR="$2"
            shift 2
            ;;
        --venv=*)
            VENV_DIR="${1#*=}"
            shift
            ;;
        --no-venv)
            USE_VENV=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

print_banner

detect_platform() {
    local uname_s
    uname_s="$(uname -s 2>/dev/null || printf unknown)"
    case "$uname_s" in
        Darwin)
            printf 'macos'
            ;;
        Linux)
            if [ -n "${ANDROID_ROOT:-}" ] || [ -n "${TERMUX_VERSION:-}" ] || [ -d /system/app ]; then
                printf 'android'
            else
                printf 'linux'
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            printf 'windows'
            ;;
        *)
            printf '%s' "$(lower "$uname_s")"
            ;;
    esac
}

PLATFORM="$(detect_platform)"

find_python_cmd() {
    PYTHON_CMD=()

    if [ -n "$PYTHON_OVERRIDE" ]; then
        PYTHON_CMD=("$PYTHON_OVERRIDE")
        return 0
    fi

    if [ "$PLATFORM" = "windows" ] && command -v py >/dev/null 2>&1; then
        PYTHON_CMD=("py" "-3")
        return 0
    fi

    local candidate
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_CMD=("$candidate")
            return 0
        fi
    done

    if [ "$PLATFORM" = "android" ] && command -v pkg >/dev/null 2>&1; then
        warn "Python was not found; trying Termux package install."
        run pkg install -y python
        if command -v python >/dev/null 2>&1; then
            PYTHON_CMD=("python")
            return 0
        fi
    fi

    die "Python 3.10 or newer was not found. Install Python first, then rerun setup."
}

check_python_version() {
    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi

    "${PYTHON_CMD[@]}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python 3.10 or newer is required; found {sys.version.split()[0]}"
    )
PY
}

create_venv() {
    if [ "$USE_VENV" -eq 0 ]; then
        PY=("${PYTHON_CMD[@]}")
        return 0
    fi

    if [ ! -d "$VENV_DIR" ]; then
        local venv_args
        venv_args=("-m" "venv")
        if [ "$PLATFORM" = "android" ]; then
            venv_args+=("--system-site-packages")
        fi
        venv_args+=("$VENV_DIR")
        run "${PYTHON_CMD[@]}" "${venv_args[@]}"
    else
        log "Using existing virtual environment: $VENV_DIR"
    fi

    if [ "$PLATFORM" = "windows" ]; then
        PY=("$VENV_DIR/Scripts/python.exe")
    else
        PY=("$VENV_DIR/bin/python")
    fi

    if [ "$DRY_RUN" -eq 0 ] && [ ! -x "${PY[0]}" ]; then
        die "virtual environment Python was not found at ${PY[0]}"
    fi
}

pip_install() {
    run "${PY[@]}" -m pip install "$@"
}

detect_cuda_version() {
    if [ "$CUDA_VERSION" != "auto" ] && [ -n "$CUDA_VERSION" ]; then
        printf '%s' "$CUDA_VERSION"
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local smi_version
        smi_version="$(
            nvidia-smi 2>/dev/null |
                sed -nE 's/.*CUDA Version:[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/p' |
                head -n 1
        )"
        if [ -n "$smi_version" ]; then
            printf '%s' "$smi_version"
            return 0
        fi
    fi

    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_version
        nvcc_version="$(
            nvcc --version 2>/dev/null |
                sed -nE 's/.*release[[:space:]]+([0-9]+(\.[0-9]+)?).*/\1/p' |
                head -n 1
        )"
        if [ -n "$nvcc_version" ]; then
            printf '%s' "$nvcc_version"
            return 0
        fi
    fi
}

version_number() {
    awk -v version="$1" 'BEGIN {
        split(version, parts, ".")
        major = parts[1] + 0
        minor = parts[2] + 0
        printf "%d", major * 100 + minor
    }'
}

select_cuda_tag() {
    local cuda_version="$1"
    local numeric
    numeric="$(version_number "$cuda_version")"

    if [ "$numeric" -ge 1300 ]; then
        printf 'cu130'
    elif [ "$numeric" -ge 1208 ]; then
        printf 'cu128'
    elif [ "$numeric" -ge 1206 ]; then
        printf 'cu126'
    elif [ "$numeric" -ge 1108 ]; then
        printf 'cu118'
    fi
}

normalize_torch_target() {
    local requested
    requested="$(lower "$TORCH_TARGET")"
    requested="${requested//./}"
    requested="${requested//-/_}"

    case "$requested" in
        auto)
            case "$PLATFORM" in
                macos)
                    printf 'mps'
                    ;;
                android)
                    printf 'cpu'
                    ;;
                *)
                    local detected_cuda selected
                    detected_cuda="$(detect_cuda_version || true)"
                    if [ -n "$detected_cuda" ]; then
                        selected="$(select_cuda_tag "$detected_cuda")"
                        if [ -n "$selected" ]; then
                            printf '%s' "$selected"
                            return 0
                        fi
                        warn "CUDA $detected_cuda was detected, but supported PyTorch wheel tags are: $SUPPORTED_CUDA_TAGS. Falling back to CPU."
                    fi
                    printf 'cpu'
                    ;;
            esac
            ;;
        cuda)
            local requested_cuda selected
            requested_cuda="$(detect_cuda_version || true)"
            [ -n "$requested_cuda" ] || die "CUDA was requested but no CUDA version could be detected. Use --cuda 12.8 or choose --cpu."
            selected="$(select_cuda_tag "$requested_cuda")"
            [ -n "$selected" ] || die "CUDA $requested_cuda is below the supported wheel floor. Use --cpu or install a newer NVIDIA driver."
            printf '%s' "$selected"
            ;;
        cpu|mps|cu118|cu126|cu128|cu130)
            printf '%s' "$requested"
            ;;
        118|11_8|cu11_8|cuda118|cuda_118|cuda11_8)
            printf 'cu118'
            ;;
        126|12_6|cu12_6|cuda126|cuda_126|cuda12_6)
            printf 'cu126'
            ;;
        128|12_8|cu12_8|cuda128|cuda_128|cuda12_8)
            printf 'cu128'
            ;;
        130|13_0|cu13_0|cuda130|cuda_130|cuda13_0)
            printf 'cu130'
            ;;
        *)
            die "unsupported torch target '$TORCH_TARGET'. Use auto, cpu, mps, cuda, cu118, cu126, cu128, or cu130."
            ;;
    esac
}

torch_index_url() {
    case "$1" in
        cu118)
            printf 'https://download.pytorch.org/whl/cu118'
            ;;
        cu126)
            printf 'https://download.pytorch.org/whl/cu126'
            ;;
        cu128)
            printf 'https://download.pytorch.org/whl/cu128'
            ;;
        cu130)
            printf 'https://download.pytorch.org/whl/cu130'
            ;;
        cpu)
            if [ "$PLATFORM" != "macos" ]; then
                printf 'https://download.pytorch.org/whl/cpu'
            fi
            ;;
    esac
}

validate_torch_variant() {
    case "$1" in
        cu*)
            if [ "$PLATFORM" = "macos" ] || [ "$PLATFORM" = "android" ]; then
                die "CUDA PyTorch wheels are not available for $PLATFORM. Use --cpu or --torch auto."
            fi
            ;;
        mps)
            if [ "$PLATFORM" != "macos" ]; then
                warn "MPS is macOS-only; using the platform default PyTorch wheel."
            fi
            ;;
    esac
}

python_imports_torch() {
    if [ "$DRY_RUN" -eq 1 ]; then
        return 1
    fi
    "${PY[@]}" - <<'PY' >/dev/null 2>&1
import torch
PY
}

install_torch() {
    local variant="$1"

    validate_torch_variant "$variant"

    if [ "$PLATFORM" = "android" ]; then
        if python_imports_torch; then
            log "Using existing Android torch install."
            return 0
        fi
        if command -v pkg >/dev/null 2>&1; then
            warn "Trying Termux python-pytorch package before pip CPU wheels."
            if run pkg install -y python-pytorch; then
                if python_imports_torch; then
                    log "Using Termux python-pytorch."
                    return 0
                fi
            fi
        fi
        warn "Falling back to pip CPU PyTorch; this may fail on Android if no compatible wheel exists."
    fi

    local index_url
    index_url="$(torch_index_url "$variant")"

    local packages
    packages=("torch" "torchvision" "torchaudio")

    if [ -n "$index_url" ]; then
        pip_install --upgrade "${packages[@]}" --index-url "$index_url"
    else
        pip_install --upgrade "${packages[@]}"
    fi
}

installed_torch_base_version() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%s' "${TC_TORCH_VERSION:-2.10.0}"
        return 0
    fi
    "${PY[@]}" - <<'PY'
import torch
print(torch.__version__.split("+", 1)[0])
PY
}

installed_torch_local_tag() {
    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi
    "${PY[@]}" - <<'PY'
import torch
version = torch.__version__
print(version.split("+", 1)[1] if "+" in version else "")
PY
}

install_project() {
    local variant="$1"
    local torch_version="$2"
    local spec="."

    if [ "$INSTALL_DEV" -eq 1 ]; then
        spec=".[dev]"
    fi

    log "Installing Cheetah with TC_TORCH_VARIANT=$variant and TC_TORCH_VERSION=$torch_version"
    if [ "$DRY_RUN" -eq 1 ]; then
        log "+ TC_TORCH_VARIANT=$variant TC_TORCH_VERSION=$torch_version ${PY[*]} -m pip install -e $spec"
        return 0
    fi

    TC_TORCH_VARIANT="$variant" TC_TORCH_VERSION="$torch_version" "${PY[@]}" -m pip install -e "$spec"
}

verify_install() {
    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi

    "${PY[@]}" - <<'PY'
import cheetah
import torch

print(f"cheetah import: ok")
print(f"torch: {torch.__version__}")
if hasattr(torch, "cuda"):
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
mps = getattr(getattr(torch, "backends", None), "mps", None)
if mps is not None:
    print(f"torch.backends.mps.is_available: {mps.is_available()}")
PY
}

find_python_cmd
check_python_version
create_venv

log "Platform: $PLATFORM"
log "Python: ${PY[*]}"

TORCH_VARIANT="$(normalize_torch_target)"
log "Selected PyTorch target: $TORCH_VARIANT"
validate_torch_variant "$TORCH_VARIANT"

pip_install --upgrade pip setuptools wheel
install_torch "$TORCH_VARIANT"
TORCH_VERSION="$(installed_torch_base_version)"
LOCAL_TAG="$(installed_torch_local_tag)"

if [[ "$TORCH_VARIANT" == cu* ]] && [ -n "$LOCAL_TAG" ] && [ "$LOCAL_TAG" != "$TORCH_VARIANT" ]; then
    warn "Installed torch local tag is '$LOCAL_TAG', but '$TORCH_VARIANT' was selected."
fi

install_project "$TORCH_VARIANT" "$TORCH_VERSION"
verify_install

log "Setup complete."
if [ "$USE_VENV" -eq 1 ]; then
    if [ "$PLATFORM" = "windows" ]; then
        log "Activate with: $VENV_DIR/Scripts/activate"
    else
        log "Activate with: source $VENV_DIR/bin/activate"
    fi
fi
