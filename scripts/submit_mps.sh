#!/bin/bash
# Usage: bash preprocess.sh $name (where $name.vrs is the vrs file to preprocess)

name=$1
aria_mps single --force --no-ui -i $name.vrs -u $ARIA_MPS_UNAME -p $ARIA_MPS_PASSW --features SLAM HAND_TRACKING
mv $name.vrs "mps_${name}_vrs/sample.vrs"
