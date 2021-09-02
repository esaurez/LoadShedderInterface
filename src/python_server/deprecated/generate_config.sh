#!/bin/bash

export bin_size=$1
export feature_bin_size=$2

source /dev/stdin <<<"$(echo 'cat <<EOF >configs/config_$1_$2.properties'; cat config.properties; echo EOF;)"
