#########################################################################
# File Name: ground.sh
# Author: Xiao Junbin
# mail: xiaojunbin@u.nus.edu
# Created Time: Mon 18 Nov 2019 03:37:25 PM +08
#########################################################################
#!/bin/bash
GPU=$1
MODE=$2
CUDA_VISIBLE_DEVICES=$GPU python ground.py --mode $MODE
