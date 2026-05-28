#!/usr/bin/env bash
# Provision a fresh Ubuntu 22.04+ VM for Auspex personal-use deployment.
#
# Run this AS the deploy user (the one whose SSH key is registered with
# the CI deploy job). The script is idempotent — safe to re-run.
#
#   curl -fsSL https://raw.githubusercontent.com/ceesartech/auspex/main/scripts/provision_vm.sh | bash
#   # or after cloning:
#   sudo -E bash scripts/provision_vm.sh
#
# What it does:
#   1. Installs Docker engine + compose plugin
#   2. Adds the current user to the docker group
#   3. Clones the auspex repo into /opt/auspex
#   4. Drops a templated .env (you must fill in secrets before `up`)
#   5. Creates /models and /data directories with right perms
#   6. Configures UFW: 22, 80, 443 open; everything else closed
#   7. Installs unattended-upgrades for security patches
#
# Tested on:
#   - Oracle Cloud free tier (Ubuntu 22.04, ARM64 A1.Flex 4OCPU/24GB)
#   - Hetzner Cloud CX22 (Ubuntu 22.04, AMD64 2vCPU/4GB)

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ceesartech/auspex.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/auspex}"
# When the script is invoked via `sudo`, $(whoami) returns "root" — which is
# never the user we want to own files or join the docker group. Prefer the
# invoking user from $SUDO_USER, falling back to whoami only when not under sudo.
TARGET_USER="${TARGET_USER:-${SUDO_USER:-$(whoami)}}"
if [ "$TARGET_USER" = "root" ]; then
  printf '\033[1;31m[provision]\033[0m %s\n' \
    "Refusing to provision as root. Re-run as a non-root user with sudo, e.g.:" >&2
  printf '\033[1;31m[provision]\033[0m %s\n' \
    "  ssh auspex@<host> 'curl -fsSL <script-url> | sudo -E bash'" >&2
  exit 1
fi

log() { printf '\033[1;34m[provision]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[provision]\033[0m %s\n' "$*" >&2; }

require_sudo() {
  if [ "$(id -u)" -ne 0 ] && ! sudo -n true 2>/dev/null; then
    err "This script needs sudo. Re-run with: sudo -E bash $0"
    exit 1
  fi
}

ensure_packages() {
  log "Updating apt and installing base packages..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    ca-certificates curl gnupg lsb-release \
    git ufw unattended-upgrades \
    htop iotop ncdu jq \
    python3 python3-venv python3-pip
}

ensure_docker() {
  if command -v docker >/dev/null 2>&1; then
    log "Docker already installed: $(docker --version)"
  else
    log "Installing Docker..."
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
      sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
      docker-ce docker-ce-cli containerd.io \
      docker-buildx-plugin docker-compose-plugin
  fi
  if ! id -nG "$TARGET_USER" | grep -qw docker; then
    log "Adding $TARGET_USER to docker group (re-login required for effect)..."
    sudo usermod -aG docker "$TARGET_USER"
  fi
  sudo systemctl enable --now docker
}

ensure_repo() {
  if [ -d "$INSTALL_DIR/.git" ]; then
    log "Repo already at $INSTALL_DIR; pulling latest..."
    sudo git -C "$INSTALL_DIR" fetch --depth 1 origin main
    sudo git -C "$INSTALL_DIR" reset --hard origin/main
  else
    log "Cloning $REPO_URL -> $INSTALL_DIR..."
    sudo mkdir -p "$INSTALL_DIR"
    sudo git clone --depth 50 "$REPO_URL" "$INSTALL_DIR"
  fi
  sudo chown -R "$TARGET_USER":"$TARGET_USER" "$INSTALL_DIR"
}

ensure_dirs() {
  log "Creating /models, /data, /var/log/auspex..."
  sudo mkdir -p /models/production /models/staging /data /var/log/auspex
  sudo chown -R "$TARGET_USER":"$TARGET_USER" /models /data /var/log/auspex
}

ensure_env_file() {
  local env_file="$INSTALL_DIR/.env"
  if [ -f "$env_file" ]; then
    log ".env already exists; leaving it alone."
    return
  fi
  log "Seeding $env_file from .env.example. Fill in secrets before starting!"
  cp "$INSTALL_DIR/.env.example" "$env_file"
  cat >> "$env_file" <<'EOF'

# --- Added by provision_vm.sh ---
# Domain Caddy will issue TLS for (set this before bringing up the prod overlay)
AUSPEX_DOMAIN=
AUSPEX_ACME_EMAIL=
EOF
  chmod 600 "$env_file"
}

ensure_firewall() {
  log "Configuring UFW..."
  sudo ufw --force reset >/dev/null
  sudo ufw default deny incoming
  sudo ufw default allow outgoing
  sudo ufw allow 22/tcp comment 'SSH'
  sudo ufw allow 80/tcp comment 'HTTP (Caddy ACME challenge)'
  sudo ufw allow 443/tcp comment 'HTTPS'
  sudo ufw --force enable
}

ensure_swap() {
  if [ "$(swapon --show=NAME | wc -l)" -gt 0 ]; then
    log "Swap already configured."
    return
  fi
  local mem_gb
  mem_gb=$(awk '/MemTotal/ {print int($2/1024/1024+0.5)}' /proc/meminfo)
  if [ "$mem_gb" -ge 8 ]; then
    log "RAM is ${mem_gb}GB; skipping swap."
    return
  fi
  log "Creating 4GB swap file (RAM=${mem_gb}GB)..."
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
}

ensure_unattended_upgrades() {
  log "Enabling unattended security upgrades..."
  # dpkg-reconfigure can hang waiting on a debconf prompt that the
  # >/dev/null redirection hides. Force the non-interactive frontend
  # and pre-seed the package, then make sure the timers are enabled.
  echo 'unattended-upgrades unattended-upgrades/enable_auto_updates boolean true' \
    | sudo debconf-set-selections
  sudo DEBIAN_FRONTEND=noninteractive dpkg-reconfigure \
    -f noninteractive unattended-upgrades >/dev/null 2>&1 || true
  sudo systemctl enable --now unattended-upgrades.service >/dev/null 2>&1 || true
  sudo systemctl enable --now apt-daily.timer apt-daily-upgrade.timer >/dev/null 2>&1 || true
}

print_next_steps() {
  cat <<EOF

==========================================================
  Provisioning complete. Next steps:

  1. Log out and back in so docker-group changes take effect
     (or run: newgrp docker).

  2. Edit secrets in $INSTALL_DIR/.env:
       cd $INSTALL_DIR
       \$EDITOR .env
     Generate values with:
       python3 -c "import secrets; print(secrets.token_hex(32))"
       python3 -c "import secrets; print(secrets.token_hex(64))"
       python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

  3. Point your domain's A record at this VM's public IP.

  4. Start the stack:
       cd $INSTALL_DIR
       docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

  5. Load historical training data (one-time):
       docker compose exec api python /app/scripts/load_football_data.py \\
         --leagues E0,D1,I1,SP1,F1 --seasons 10

  6. Train initial models:
       docker compose exec api python -m training.train_all_models \\
         --output-dir /models/staging --export-onnx

  7. Verify https://YOUR_DOMAIN/health returns {"status":"healthy"}.

  For CI auto-deploys, set in the GitHub repo:
    vars.DEPLOY_ENABLED=true
    vars.DEPLOY_HOST=$(curl -s ifconfig.me 2>/dev/null || echo '<vm-ip>')
    vars.DEPLOY_USER=$TARGET_USER
    secrets.DEPLOY_SSH_KEY=<contents of ~/.ssh/id_ed25519 from a workstation>
==========================================================
EOF
}

main() {
  require_sudo
  ensure_packages
  ensure_docker
  ensure_repo
  ensure_dirs
  ensure_env_file
  ensure_swap
  ensure_firewall
  ensure_unattended_upgrades
  print_next_steps
}

main "$@"
