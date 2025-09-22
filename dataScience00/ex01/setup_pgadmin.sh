#!/bin/bash

# exit on error
set -e

# setup dirs
sudo mkdir -p /var/lib/pgadmin/sessions
sudo mkdir -p /var/lib/pgadmin/storage
sudo chmod -R 777 /var/lib/pgadmin/
sudo chmod -R 777 /var/log/pgadmin/

# create venv
python3 -m venv ~/pgadmin4-venv
source ~/pgadmin4-venv/bin/activate

# Install pgAdmin 4
pip install pgadmin4

# Setup using the pgadmin4 CLI command
~/pgadmin4-venv/bin/pgadmin4 setup --email "ssian@42mail.sutd.edu.sg" --password "ssian12345"

# Launch pgAdmin 4
#~/pgadmin4-venv/bin/pgadmin4
