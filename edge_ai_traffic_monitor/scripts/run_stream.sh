#!/bin/bash
# Quick-launch browser-based detection stream
cd "$(dirname "$0")/.."
echo "Starting detection stream..."
echo "Open http://192.168.137.96:8080 in your browser"
python3 deploy/python/stream_server.py