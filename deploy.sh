#!/usr/bin/env bash
# Build and deploy to https://pikos.org/onni/water/
set -euo pipefail
cd "$(dirname "$0")"
npm run build
rsync -az --delete -e "ssh -p 2323" dist/ onni@pikos.org:/srv/www/onni/water/
echo "Deployed: https://pikos.org/onni/water/"
