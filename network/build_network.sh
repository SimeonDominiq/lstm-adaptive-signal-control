#!/bin/bash
# Build the SUMO network from the source definitions.
# Run from the project root:  bash network/build_network.sh
set -e
netconvert \
    --node-files=network/nodes.nod.xml \
    --edge-files=network/edges.edg.xml \
    --connection-files=network/connections.con.xml \
    --output-file=network/intersection.net.xml \
    --no-turnarounds true \
    --tls.default-type static \
    --default.junctions.keep-clear true
echo "network written to network/intersection.net.xml"
echo ""
echo "Now generate the signal programme from the built network:"
echo "    python3 scripts/make_tls.py"
