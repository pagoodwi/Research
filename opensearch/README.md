# OpenSearch 3.1 Tarball Deployment

## Overview
- 3 cluster-manager nodes (with data): node-1, node-2, node-3
- 1 data-only node: node-4
- All use persistent volume paths like `/mnt/pvc/opensearch/nodeX/data`

## Instructions

1. Download OpenSearch 3.1.0 tarball into the root of this folder:
   https://opensearch.org/downloads.html

2. Place file as:
   opensearch-3.1.0-linux-x64.tar.gz

3. Run bootstrap:
   ```bash
   chmod +x Bootstrap.sh
   ./Bootstrap.sh
