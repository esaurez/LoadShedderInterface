#!/bin/bash

export mode=$1
export title=$2
export mbs=$3
source /dev/stdin <<<"$(echo 'cat <<EOF >configs/hsb_config_$2.properties'; cat configs/hsb_configs_template.properties;)"
#source /dev/stdin <<<"$(echo 'cat  <<EOF >configs/hsb_config_$2.properties'; cat configs/hsb_configs_template.properties;)"
