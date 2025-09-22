#!/bin/bash

# exit on error
set -e

# Create necessary directories
mkdir -p ~/pgadmin4-data
mkdir -p ~/pgadmin4-data/sessions
mkdir -p ~/pgadmin4-data/storage

# Create config file
cat > ~/pgadmin4-data/config_local.py <<EOF
import os

SERVER_MODE = True
DATA_DIR = os.path.expanduser(os.path.join("~", "pgadmin4-data"))
SQLITE_PATH = os.path.join(DATA_DIR, 'pgadmin4.db')
SESSION_DB_PATH = os.path.join(DATA_DIR, 'sessions')
STORAGE_DIR = os.path.join(DATA_DIR, 'storage')
LOG_FILE = os.path.join(DATA_DIR, 'pgadmin4.log')
EOF

# Create virtual environment
python3 -m venv ~/pgadmin4-venv
source ~/pgadmin4-venv/bin/activate

# Install pgAdmin 4
pip install pgadmin4

# Set up initial login
export PGADMIN_SETUP_EMAIL="ssian@42mail.sutd.edu.sg"
export PGADMIN_SETUP_PASSWORD="ssian12345"

# Run the setup command first
echo "Setting up pgAdmin 4..."
python ~/pgadmin4-venv/lib/python3.*/site-packages/pgadmin4/setup.py

# Launch pgAdmin 4 (this will run in foreground)
echo "Starting pgAdmin 4..."
python ~/pgadmin4-venv/lib/python3.*/site-packages/pgadmin4/pgAdmin4.py