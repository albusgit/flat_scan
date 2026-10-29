#!/usr/bin/env bash
# setup_box.sh: on a vast.ai PyTorch instance, after box.sh and dash.py are copied to /root:
#   - the dashboard (dash.py) as a supervisor service, in the portal's "Open" menu as "FlatScan Training" (port 10100,
#     behind the portal's token);
#   - the training (box.sh) in tmux session "train", started as soon as Hugging Face is logged in on the box
#     (HF_TOKEN in the environment or `hf auth login`): the bundle is a private repo.
set -euo pipefail
PRICE=${PRICE:-1.2}

cat > /opt/supervisor-scripts/flatscan-dash.sh <<EOF
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "\${utils}/logging.sh"
. "\${utils}/environment.sh"
. "\${utils}/exit_portal.sh" "FlatScan Training"
exec /usr/bin/python3 /root/dash.py --work /root/fs/work --port 17070 --price $PRICE 2>&1
EOF
chmod +x /opt/supervisor-scripts/flatscan-dash.sh
cat > /etc/supervisor/conf.d/flatscan-dash.conf <<'EOF'
[program:flatscan-dash]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/flatscan-dash.sh
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
EOF
/venv/main/bin/python - <<'EOF'
import yaml
d = yaml.safe_load(open('/etc/portal.yaml')) or {'applications': {}}
d['applications']['FlatScan Training'] = {'hostname': 'localhost', 'external_port': 10100, 'internal_port': 17070,
                                          'open_path': '/', 'name': 'FlatScan Training'}
yaml.safe_dump(d, open('/etc/portal.yaml', 'w'), sort_keys=False)
EOF
supervisorctl reread > /dev/null && supervisorctl update > /dev/null
supervisorctl restart caddy > /dev/null
sleep 2
supervisorctl status flatscan-dash

cat > /root/launch.sh <<'EOF'
#!/usr/bin/env bash
cd /root
set +u; source /venv/main/bin/activate; set -u
[ -f /workspace/.env ] && set -a && . /workspace/.env && set +a
echo "$(date +%H:%M:%S) waiting for a Hugging Face login (HF_TOKEN, or run: hf auth login)"
until hf auth whoami > /dev/null 2>&1; do sleep 10; [ -f /workspace/.env ] && set -a && . /workspace/.env && set +a; done
echo "$(date +%H:%M:%S) logged in as $(hf auth whoami 2>/dev/null | head -1): starting box.sh"
STORE=hf:SealAItest/fs-r19 bash /root/box.sh
echo "$(date +%H:%M:%S) box.sh ended"; exec bash
EOF
chmod +x /root/launch.sh
tmux has-session -t train 2> /dev/null || tmux new-session -d -s train /root/launch.sh
tmux ls
