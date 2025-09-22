#!/bin/bash

# exit on error
set -e

# create venv
python3 -m venv ~/pgadmin4-venv
source ~/pgadmin4-venv/bin/activate

# Install pgAdmin 4
pip install pgadmin4
pip install psycopg2-binary
pip install flake8

# setup dirs
sudo mkdir -p /var/log/pgadmin
sudo mkdir -p /var/lib/pgadmin
sudo chmod -R 777 /var/log/pgadmin
sudo chmod -R 777 /var/lib/pgadmin

# Start pgadmin4
~/pgadmin4-venv/bin/pgadmin4
