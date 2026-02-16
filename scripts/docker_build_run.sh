#!/usr/bin/env zsh
set -euo pipefail

# scripts/docker_build_run.sh
#
# Build and optionally run/stop Docker images for this project with input validation.
#
# Supported build types:
#   mount-model   -> build target "mount-model" (model mounted at runtime)
#   with-model    -> build target "with-model"  (model copied into image)
#   both          -> build both targets
#
# Actions:
#   --run         -> start container(s) (default action if none specified is "build only")
#   --stop        -> stop container(s) (no build)
#
# Examples:
#   ./scripts/docker_build_run.sh --type both --run
#   ./scripts/docker_build_run.sh --type mount-model --run --port 8000
#   ./scripts/docker_build_run.sh --type with-model --run --port 8001
#   ./scripts/docker_build_run.sh --type with-model            # build only
#   ./scripts/docker_build_run.sh --type both --stop
#   ./scripts/docker_build_run.sh --type mount-model --stop
#
# Optional:
#   --image cre-underwriting-api
#   --port 8000                 # used for single-type runs
#   --port-mount 8000           # used for --type both
#   --port-with 8001            # used for --type both
#   --no-health                 # skip curl /health checks
#
# Requirements:
#   - Docker daemon running
#   - curl available if --run (and health checks not disabled)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Defaults
TYPE=""
RUN=0
STOP=0
IMAGE="cre-underwriting-api"
PORT=""
PORT_MOUNT="8000"
PORT_WITH="8001"
DO_HEALTH=1

_usage() {
  cat <<'USAGE'
Usage:
  scripts/docker_build_run.sh --type <mount-model|with-model|both> [--run|--stop]
                             [--image <name>]
                             [--port <host_port>]                  (single-type run)
                             [--port-mount <host_port>]            (both run)
                             [--port-with <host_port>]             (both run)
                             [--no-health]

Notes:
  - If neither --run nor --stop is provided, the script builds only.

Examples:
  ./scripts/docker_build_run.sh --type both --run
  ./scripts/docker_build_run.sh --type mount-model --run --port 8000
  ./scripts/docker_build_run.sh --type with-model --run --port 8001
  ./scripts/docker_build_run.sh --type with-model
  ./scripts/docker_build_run.sh --type both --stop
USAGE
}

_is_int() { [[ "${1:-}" =~ '^[0-9]+$' ]]; }

_validate_port() {
  local p="${1:-}"
  _is_int "$p" || return 1
  (( p >= 1 && p <= 65535 )) || return 1
  return 0
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --type) TYPE="${2:-}"; shift 2 ;;
    --run)  RUN=1; shift ;;
    --stop) STOP=1; shift ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --port-mount) PORT_MOUNT="${2:-}"; shift 2 ;;
    --port-with) PORT_WITH="${2:-}"; shift 2 ;;
    --no-health) DO_HEALTH=0; shift ;;
    -h|--help) _usage; exit 0 ;;
    *) echo "FAIL: unknown arg: $1" >&2; _usage; exit 2 ;;
  esac
done

if [[ -z "$TYPE" ]]; then
  echo "FAIL: --type is required" >&2
  _usage
  exit 2
fi

case "$TYPE" in
  mount-model|with-model|both) ;;
  *) echo "FAIL: --type must be one of: mount-model, with-model, both" >&2; _usage; exit 2 ;;
esac

if [[ "$RUN" -eq 1 && "$STOP" -eq 1 ]]; then
  echo "FAIL: choose only one action: --run or --stop" >&2
  exit 2
fi

# Require docker
command -v docker >/dev/null || { echo "FAIL: docker not found" >&2; exit 1; }

# Container names
CNAME_MOUNT="${IMAGE}-mount"
CNAME_WITH="${IMAGE}-with"

_stop_container() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    docker rm -f "$name" >/dev/null 2>&1 || true
    echo "OK: stopped $name"
  else
    echo "OK: not running $name"
  fi
}

_health() {
  local p="$1"
  if [[ "$DO_HEALTH" -eq 0 ]]; then
    return 0
  fi
  command -v curl >/dev/null || { echo "FAIL: curl not found (needed for health check)" >&2; exit 1; }
  curl -sf "http://127.0.0.1:${p}/health" >/dev/null
}

# STOP action (no build)
if [[ "$STOP" -eq 1 ]]; then
  echo "== docker: stop =="
  case "$TYPE" in
    mount-model) _stop_container "$CNAME_MOUNT" ;;
    with-model)  _stop_container "$CNAME_WITH" ;;
    both)
      _stop_container "$CNAME_MOUNT"
      _stop_container "$CNAME_WITH"
      ;;
  esac
  echo "OK: stop complete"
  exit 0
fi

# Validate ports if running
if [[ "$RUN" -eq 1 ]]; then
  if [[ "$TYPE" == "both" ]]; then
    _validate_port "$PORT_MOUNT" || { echo "FAIL: --port-mount must be 1..65535" >&2; exit 2; }
    _validate_port "$PORT_WITH"  || { echo "FAIL: --port-with must be 1..65535" >&2; exit 2; }
    if [[ "$PORT_MOUNT" == "$PORT_WITH" ]]; then
      echo "FAIL: --port-mount and --port-with must be different" >&2
      exit 2
    fi
  else
    if [[ -z "$PORT" ]]; then
      PORT="8000"
      [[ "$TYPE" == "with-model" ]] && PORT="8001"
    fi
    _validate_port "$PORT" || { echo "FAIL: --port must be 1..65535" >&2; exit 2; }
  fi
fi

# Build
if [[ "$TYPE" == "mount-model" || "$TYPE" == "both" ]]; then
  echo "== docker: build (mount-model) =="
  docker build --target mount-model -t "${IMAGE}:mount" .
fi

if [[ "$TYPE" == "with-model" || "$TYPE" == "both" ]]; then
  echo "== docker: build (with-model) =="
  docker build --target with-model -t "${IMAGE}:with-model" .
fi

# Build-only default
if [[ "$RUN" -ne 1 ]]; then
  echo "OK: build complete (no --run)"
  exit 0
fi

# Run
case "$TYPE" in
  mount-model)
    if [[ ! -f "models/model.joblib" ]]; then
      echo "FAIL: models/model.joblib not found (required for mount-model run). Run: python -m src.train" >&2
      exit 1
    fi
    _stop_container "$CNAME_MOUNT" >/dev/null 2>&1 || true
    echo "== docker: run (mount-model) on :${PORT} =="
    docker run -d --rm \
      --name "$CNAME_MOUNT" \
      -p "${PORT}:8000" \
      -v "$PWD/models:/app/models" \
      "${IMAGE}:mount" >/dev/null
    echo "== docker: health =="
    _health "$PORT" && echo "OK: http://127.0.0.1:${PORT}/health"
    echo "OK: running ${CNAME_MOUNT} on http://127.0.0.1:${PORT}"
    ;;
  with-model)
    _stop_container "$CNAME_WITH" >/dev/null 2>&1 || true
    echo "== docker: run (with-model) on :${PORT} =="
    docker run -d --rm \
      --name "$CNAME_WITH" \
      -p "${PORT}:8000" \
      "${IMAGE}:with-model" >/dev/null
    echo "== docker: health =="
    _health "$PORT" && echo "OK: http://127.0.0.1:${PORT}/health"
    echo "OK: running ${CNAME_WITH} on http://127.0.0.1:${PORT}"
    ;;
  both)
    if [[ ! -f "models/model.joblib" ]]; then
      echo "FAIL: models/model.joblib not found (required for mount-model run). Run: python -m src.train" >&2
      exit 1
    fi
    _stop_container "$CNAME_MOUNT" >/dev/null 2>&1 || true
    _stop_container "$CNAME_WITH"  >/dev/null 2>&1 || true

    echo "== docker: run (mount-model) on :${PORT_MOUNT} =="
    docker run -d --rm \
      --name "$CNAME_MOUNT" \
      -p "${PORT_MOUNT}:8000" \
      -v "$PWD/models:/app/models" \
      "${IMAGE}:mount" >/dev/null

    echo "== docker: run (with-model) on :${PORT_WITH} =="
    docker run -d --rm \
      --name "$CNAME_WITH" \
      -p "${PORT_WITH}:8000" \
      "${IMAGE}:with-model" >/dev/null

    echo "== docker: health =="
    _health "$PORT_MOUNT" && echo "OK: http://127.0.0.1:${PORT_MOUNT}/health (mount-model)"
    _health "$PORT_WITH"  && echo "OK: http://127.0.0.1:${PORT_WITH}/health (with-model)"

    echo "OK: running"
    echo "  mount-model: http://127.0.0.1:${PORT_MOUNT}  (container: ${CNAME_MOUNT})"
    echo "  with-model : http://127.0.0.1:${PORT_WITH}  (container: ${CNAME_WITH})"
    ;;
esac

echo "Stop: $0 --type ${TYPE} --stop --image ${IMAGE}"