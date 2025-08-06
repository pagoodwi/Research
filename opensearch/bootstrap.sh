#!/bin/bash
echo "Bootstrapping OpenSearch nodes..."

for i in 1 2 3 4; do
  echo "Setting up node$i"
  mkdir -p node$i
  tar -xzf opensearch-3.1.0-linux-x64.tar.gz -C node$i
  cp OpenSearchNode$i.yaml node$i/opensearch-3.1.0/config/opensearch.yml
done

echo "Done. You can now start each node manually."
