# Deploying river_forecast on a VPS

The forecast runs as a one-shot Docker container triggered every 15 minutes by a
systemd timer. The container writes `forecast.png` to `/var/www/river_forecast`,
which nginx serves on localhost only. A Cloudflare Tunnel publishes it at a
hostname of your choice — the server needs **no open inbound ports** (the tunnel
makes an outbound-only connection). No API key is needed for the flow data: it
comes from the public Hub'Eau API (French government).

Placeholders used below: `YOUR_HOSTNAME` (e.g. `forecast.example.com`) and
`YOUR_TUNNEL` (a name for the tunnel, e.g. `river-forecast`).

## 1. Forecast generation (Docker + systemd timer)

```bash
# Install Docker (Debian/Ubuntu)
sudo apt update && sudo apt install -y docker.io

# Clone and build
git clone https://github.com/maxnolte/river_forecast.git
cd river_forecast
sudo docker build -t river-forecast:py37 .

# Output directory
sudo mkdir -p /var/www/river_forecast

# Test one run
sudo docker run --rm -v /var/www/river_forecast:/output river-forecast:py37
ls -l /var/www/river_forecast/forecast.png

# Install the schedule
sudo cp deploy/river-forecast.service deploy/river-forecast.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now river-forecast.timer
```

## 2. Local web server (nginx)

```bash
sudo apt install -y nginx
# Edit server_name (YOUR_HOSTNAME) first
sudo cp deploy/nginx-river-forecast.conf /etc/nginx/sites-available/river-forecast
sudo ln -s /etc/nginx/sites-available/river-forecast /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

The provided config listens on `127.0.0.1:80` only, so nginx is not reachable
from the internet even without a firewall. Verify locally:

```bash
curl -sI http://localhost/forecast.png
```

## 3. Publish via Cloudflare Tunnel

Requires a domain whose DNS is managed by Cloudflare.

```bash
# Install cloudflared (Debian/Ubuntu; see Cloudflare docs for other systems)
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared

# Authorize against your Cloudflare account (opens a browser URL)
cloudflared tunnel login

# Create the tunnel and route a hostname to it
cloudflared tunnel create YOUR_TUNNEL
cloudflared tunnel route dns YOUR_TUNNEL YOUR_HOSTNAME
```

Create `/etc/cloudflared/config.yml` (the tunnel UUID and credentials path are
printed by `tunnel create`):

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: /home/<user>/.cloudflared/<TUNNEL_UUID>.json
ingress:
  - hostname: YOUR_HOSTNAME
    service: http://localhost:80
  - service: http_status:404
```

Then run it as a service:

```bash
sudo cloudflared --config /etc/cloudflared/config.yml service install
sudo systemctl enable --now cloudflared
```

The forecast is then available at `https://YOUR_HOSTNAME/forecast.png`.

Recommended: in the Cloudflare dashboard, add a Cache Rule for `YOUR_HOSTNAME`
with Browser TTL set to "Respect origin" — nginx sends `no-cache` so the image
(regenerated every 15 minutes) is never served stale, while Cloudflare's edge
still absorbs the traffic via revalidation.

## Operations

```bash
systemctl list-timers river-forecast.timer   # next/last run
journalctl -u river-forecast.service -n 50   # forecast logs
journalctl -u cloudflared -n 50              # tunnel logs
sudo systemctl start river-forecast.service  # trigger a run manually
```

After updating the code: `git pull`, rebuild the image, done — the next timer
tick uses the new image.
