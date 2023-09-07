#!/bin/bash

python app.py -c configs/run_continuous_maria.txt --script-mode=viewer \
  --dataset-render-poses-centeroffset 0.9 0.0 0.0
