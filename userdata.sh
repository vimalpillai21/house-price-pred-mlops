#!/bin/bash

set -e
export APP_DIR=/opt/house-price-app
mkdir -p $APP_DIR

apt install -y
apt install -y git nginx python3 python3-venv python3-pip
curl -LsSf https://astral.sh/uv/install.sh | sh

cd $APP_DIR
git clone https://github.com/vimalpillai21/house-price-pred-mlops .
git switch asg-deployment
uv sync
source .venv/bin/activate

# Configure uvicorn systemd service
cat > /etc/systemd/system/house_uvicorn.service << 'EOF'
[Unit]
Description=Uvicorn instance for House Price Regressor
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/house-price-app
Environment="PATH=/opt/house-price-app/.venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/path/to/venv/bin/uvicorn api.inference_apis:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# configure nginx reverse proxy
cat >/etc/nginx/conf.d/house_app.conf << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 60s;
        proxy_read_timeout 120s;
    }
}
EOF

# Remove default site if present to avoid conflict
if [ -L /etc/nginx/sites-enabled/default ] || [ -f /etc/nginx/sites-enabled/default ]; then
  rm -f /etc/nginx/sites-enabled/default || true
fi

# start and enable services
systemctl daemon-reload
systemctl enable house_app
systemctl start house_app
systemctl enable nginx
systemctl start nginx