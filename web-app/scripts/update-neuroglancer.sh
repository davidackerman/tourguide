#!/usr/bin/env bash
# Bump neuroglancer to the latest release on npm and verify the build still
# passes. Run from the web-app/ directory:
#
#   ./scripts/update-neuroglancer.sh
#
# After it runs, commit the package.json + package-lock.json changes.

set -euo pipefail

cd "$(dirname "$0")/.."

OLD_VERSION=$(node -p "require('./package.json').dependencies.neuroglancer")
echo "Current: neuroglancer@${OLD_VERSION}"

npm install neuroglancer@latest

NEW_VERSION=$(node -p "require('./package.json').dependencies.neuroglancer")
echo "Updated:  neuroglancer@${NEW_VERSION}"

if [ "$OLD_VERSION" = "$NEW_VERSION" ]; then
  echo "Already on latest. Nothing to do."
  exit 0
fi

echo
echo "Verifying build…"
npm run build

echo
echo "✓ Built clean on neuroglancer@${NEW_VERSION}."
echo "Review and commit:"
echo "  git diff package.json package-lock.json"
echo "  git commit -am 'deps(web-app): bump neuroglancer ${OLD_VERSION} -> ${NEW_VERSION}'"
