#!/bin/bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  mlops-pipeline python app/main.py

