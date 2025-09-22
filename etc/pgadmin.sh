#!/bin/bash

# exit on error
set -e

mkdir -p ~/pgadmin4-data
mkdir -p ~/pgadmin4-data/sessions
mkdir -p ~/pgadmin4-data/storage

cat > ~/pgadmin4-data/config_local.py <<EOF
import os

SERVER_MODE = True
DATA_DIR = os.path.expanduser(os.path.join("~", "pgadmin4-data"))
SQLITE_PATH = os.path.join(DATA_DIR, 'pgadmin4.db')
#SESSION_DB_PATH = os.path.join(DATA_DIR, 'sessions')
SESSION_DB_PATH = os.path.join(DATA_DIR, 'sessions', 'pgadmin4.db') 
STORAGE_DIR = os.path.join(DATA_DIR, 'storage')
LOG_FILE = os.path.join(DATA_DIR, 'pgadmin4.log')
EOF

python3 -m venv ~/pgadmin4-venv
source ~/pgadmin4-venv/bin/activate

# Install pgAdmin 4
pip install pgadmin4

# Set up initial login
export PGADMIN_SETUP_EMAIL="ssian@42mail.sutd.edu.sg"
export PGADMIN_SETUP_PASSWORD="ssian12345"

#export PGADMIN_CONFIG_HOME=~/pgadmin4-data
#export PGADMIN_CONFIG_LOCAL_FILE=~/pgadmin4-data/config_local.py
export PGADMIN_CONFIG_FILE=~/pgadmin4-data/config_local.py
#export PGADMIN_CONFIG_HOME=~/pgadmin4-data/config_local.py

export PGADMIN_CONFIG_SERVER_MODE=True

# Launch pgAdmin 4
~/pgadmin4-venv/bin/pgadmin4

