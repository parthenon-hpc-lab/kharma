#!/bin/bash

export PROJ_NAME=iris

KHARMA_DIR=$(dirname "$(readlink -f "$0")")
$KHARMA_DIR/run.sh "$@"
