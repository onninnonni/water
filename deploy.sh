#!/usr/bin/env bash
# Build and deploy to https://onnin.no (files in /srv/www/onni/onnin, hidden from pikos.org)
set -euo pipefail
cd "$(dirname "$0")"
npm run build
rsync -az --delete -e "ssh -p 2323" dist/ onni@pikos.org:/srv/www/onni/onnin/
echo "Deployed: https://onnin.no"
