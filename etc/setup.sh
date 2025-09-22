#!/bin/bash
# Exit immediately if a command fails
set -e

# ---------------------------
# 1. Create directories
# ---------------------------
PGADMIN_DATA_DIR=~/pgadmin4-data
mkdir -p "$PGADMIN_DATA_DIR/sessions"
mkdir -p "$PGADMIN_DATA_DIR/storage"

# ---------------------------
# 2. Create local config
# ---------------------------
cat > "$PGADMIN_DATA_DIR/config_local.py" <<EOF
import os

DATA_DIR = os.path.expanduser(os.path.join("~", "pgadmin4-data"))
SQLITE_PATH = os.path.join(DATA_DIR, 'pgadmin4.db')
SESSION_DB_PATH = os.path.join(DATA_DIR, 'sessions')
STORAGE_DIR = os.path.join(DATA_DIR, 'storage')
LOG_FILE = os.path.join(DATA_DIR, 'pgadmin4.log')

# Force server mode
SERVER_MODE = True
EOF

# ---------------------------
# 3. Set up Python virtual environment
# ---------------------------
PYENV_DIR=~/pgadmin4-venv
python3 -m venv "$PYENV_DIR"
source "$PYENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install pgAdmin 4
pip install pgadmin4

# ---------------------------
# 4. Set environment variables
# ---------------------------
export PGADMIN_CONFIG_FILE="$PGADMIN_DATA_DIR/config_local.py"
export PGADMIN_SETUP_EMAIL="ssian@42mail.sutd.edu.sg"
export PGADMIN_SETUP_PASSWORD="ssian12345"

# ---------------------------
# 5. Launch pgAdmin 4 in server mode
# ---------------------------
echo "Launching pgAdmin 4..."
echo "Access it in your browser at http://127.0.0.1:5050"
"$PYENV_DIR/bin/pgadmin4"
