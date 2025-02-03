#!/bin/bash
./bazel-bin/cuda/test_cuda_blend ${HOME}/src/hm/left_ready.png ${HOME}/src/hm/right_ready.png ${HOME}/src/hm/seam_mask_ready.png ${HOME}/src/jetson-utils/output.png
