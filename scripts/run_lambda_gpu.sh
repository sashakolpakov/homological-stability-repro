#!/usr/bin/env bash
set -euo pipefail

SSH_USER="${SSH_USER:-ubuntu}"
IP="${IP:-}"
HOST="${HOST:-}"
SSH_KEY="${SSH_KEY:-}"
GPU_TASK="${GPU_TASK:-benchmarks}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_lambda_gpu.sh --ip 132.145.143.182 --ssh-key ~/.ssh/id_ed25519

This is a Lambda.ai / Lambda Labs convenience wrapper around run_remote_gpu.sh.
It prompts for missing values, prints the public SSH key to register with
Lambda if available, builds ubuntu@IP from --ip, and then delegates the actual
SSH/Docker upload, run, and download workflow to run_remote_gpu.sh.

Before booting the instance, give Lambda the public key matching the private key:
  cat ~/.ssh/id_ed25519.pub

Options:
  --ip IP_ADDRESS
  --host USER@IP_ADDRESS
  --ssh-key PATH_TO_PRIVATE_KEY
  --user SSH_USER                default: ubuntu
  --task benchmarks|topology-sweep
  --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ip)
      IP="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --ssh-key)
      SSH_KEY="$2"
      shift 2
      ;;
    --user)
      SSH_USER="$2"
      shift 2
      ;;
    --task)
      GPU_TASK="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$HOST" && -z "$IP" ]]; then
  read -r -p "GPU instance IP address or SSH login [ubuntu@IP]: " TARGET
  if [[ "$TARGET" == *@* ]]; then
    HOST="$TARGET"
  else
    IP="$TARGET"
  fi
fi

if [[ -z "$HOST" ]]; then
  HOST="$SSH_USER@$IP"
fi

if [[ -z "$SSH_KEY" ]]; then
  for candidate in "$HOME/.ssh/id_ed25519" "$PWD/../.ssh/id_ed25519"; do
    if [[ -r "$candidate" ]]; then
      SSH_KEY="$candidate"
      break
    fi
  done
fi

if [[ -z "$SSH_KEY" ]]; then
  read -r -p "Path to matching private SSH key: " SSH_KEY
fi

if [[ ! -r "$SSH_KEY" ]]; then
  echo "Private SSH key is not readable: $SSH_KEY" >&2
  exit 2
fi

PUB_KEY="${SSH_KEY}.pub"
if [[ -r "$PUB_KEY" ]]; then
  echo "Public key for new Lambda instances:"
  cat "$PUB_KEY"
  echo
fi

echo "Running remote GPU workflow on $HOST"
HOST="$HOST" SSH_KEY="$SSH_KEY" GPU_TASK="$GPU_TASK" REMOTE_DOCKER="${REMOTE_DOCKER:-auto}" DOCKER_BUILD_FLAGS="${DOCKER_BUILD_FLAGS:-}" scripts/run_remote_gpu.sh
