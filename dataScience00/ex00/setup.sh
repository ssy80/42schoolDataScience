#!/bin/bash

# exit on error
set -e

DB="piscineds"
DB_USER="ssian"
DB_USER_PASS="mysecretpassword"

# install postgres
sudo apt install -y postgresql

# enable and start postgres
sudo systemctl enable postgresql
sudo systemctl start postgresql

# login and create DB and user
# Run psql as the postgres superuser
sudo -u postgres psql <<EOF
CREATE DATABASE ${DB};
CREATE USER ${DB_USER} WITH LOGIN PASSWORD '${DB_USER_PASS}';
GRANT ALL PRIVILEGES ON DATABASE ${DB} to ${DB_USER}; 
EOF
