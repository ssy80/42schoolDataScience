#!/bin/bash
set -e

# Create virtual environment and install pgadmin4
python3 -m venv ~/pgadmin4-venv
source ~/pgadmin4-venv/bin/activate
pip install pgadmin4

# Set credentials
export PGADMIN_SETUP_EMAIL="ssian@42mail.sutd.edu.sg"
export PGADMIN_SETUP_PASSWORD="ssian12345"

# Run setup and launch
pgadmin4