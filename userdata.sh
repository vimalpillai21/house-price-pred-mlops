#!/bin/bash

set -euo pipefail

APP_USER=ubuntu
APP_DIR=/opt/house-price-app
UV=/home/$APP_USER/.local/bin/uv
export DEBIAN_FRONTEND=noninteractive

if [ "$EUID" -ne 0 ]; then
    echo "Run with: sudo bash $0"
    exit 1
fi

apt update -y
apt install -y git nginx python3 python3-venv python3-pip

mkdir -p "$APP_DIR"
chown -R $APP_USER:$APP_USER "$APP_DIR"

sudo -u $APP_USER bash -s <<EOF
set -euo pipefail

if [ ! -x "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "$APP_DIR"

if [ ! -d .git ]; then
  git clone -b asg-deployment https://github.com/vimalpillai21/house-price-pred-mlops .
else
  git fetch origin
  git checkout asg-deployment
  git pull
fi

$UV venv
$UV pip install -r ./requirements/requirements-api.txt
EOF

test -x "$APP_DIR/.venv/bin/uvicorn" || { echo "uvicorn missing in venv"; exit 1; } 


# Configure uvicorn systemd service
cat > /etc/systemd/system/house_uvicorn.service << EOF
[Unit]
Description=Uvicorn instance for House Price Regressor
After=network.target

[Service]
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/.venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=$APP_DIR/.venv/bin/uvicorn api.inference_apis:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
EOF


# Remove default site if present to avoid conflict
if [ -L /etc/nginx/sites-enabled/default ] || [ -f /etc/nginx/sites-enabled/default ]; then
  rm -f /etc/nginx/sites-available/default || true
  rm -f /etc/nginx/sites-enabled/default || true
fi

# remove stock site first
rm -f /etc/nginx/sites-enabled/default /etc/nginx/sites-available/default

cat >/etc/nginx/conf.d/house_app.conf << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 60s;
        proxy_read_timeout 120s;
    }
}
EOF

systemctl daemon-reload
systemctl enable --now house_uvicorn
systemctl restart house_uvicorn
nginx -t
systemctl enable nginx
systemctl restart nginx