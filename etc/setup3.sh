#!/bin/bash
set -e

# Create data directories
mkdir -p ~/pgadmin4-data
mkdir -p ~/pgadmin4-data/sessions
mkdir -p ~/pgadmin4-data/storage

DATA_DIR=~/pgadmin4-data
VENV_DIR=~/pgadmin4-venv

# Create local config
cat > "$DATA_DIR/config_local.py" <<EOF
import os

DATA_DIR = os.path.expanduser("~/pgadmin4-data")
#SESSION_DB_PATH = os.path.join(DATA_DIR, 'sessions')
SESSION_DB_PATH = os.path.expanduser('~/pgadmin4-data/sessions')
SQLITE_PATH = os.path.join(DATA_DIR, 'pgadmin4.db')
STORAGE_DIR = os.path.join(DATA_DIR, 'storage')
LOG_FILE = os.path.join(DATA_DIR, 'pgadmin4.log')
SERVER_MODE = True
DEFAULT_SERVER = '127.0.0.1'
DEFAULT_SERVER_PORT = 5050
EOF

# Set up virtual environment
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install pgadmin4

# Set environment variables
export PGADMIN_CONFIG_FILE="$DATA_DIR/config_local.py"
export PGADMIN_SETUP_EMAIL="ssian@42mail.sutd.edu.sg"
export PGADMIN_SETUP_PASSWORD="ssian12345"

# Setup pgAdmin4 (first-time configuration)
echo "Setting up pgAdmin4 for the first time..."
python3 -c "
import os
os.environ['PGADMIN_CONFIG_FILE'] = '$DATA_DIR/config_local.py'
from pgadmin4 import pgAdmin4
app = pgAdmin4()
app.create_app()
"

# Launch pgAdmin4 in server mode
echo "Launching pgAdmin 4 at http://127.0.0.1:5050"
echo "Login with email: $PGADMIN_SETUP_EMAIL and password: $PGADMIN_SETUP_PASSWORD"
"$VENV_DIR/bin/python" "$VENV_DIR/lib/python3.12/site-packages/pgadmin4/pgAdmin4.py"