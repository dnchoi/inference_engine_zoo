#!/bin/sh

# commend
# bash build_run.sh build 1 3 112 112 10 1 0 /mnt/md0/FAIP.3.0/candidate_model/model.onnx onnx img.png
# exit on first error
set -e

BUILD=$1
BATCH=$2
C=$3
W=$4
H=$5
ITER=$6
ACC=$7
OPTI=$8
MODEL=$9
ENGINE=${10}
IMG=${11}

echo "BUILD or RUN : $BUILD"
echo "INPUT BATCH SIZE : $BATCH"
echo "INPUT CHANNEL : $C"
echo "INPUT WIDTH : $W"
echo "INPUT HEIGHT : $H"
echo "ITERATE : $ITER"
echo "ACCELERATOR : $ACC"
echo "OPTIMIZER : $OPTI"
echo "MODEL PATH : $MODEL"
echo "USING ENGINE : $ENGINE"
echo "IMAGE PATH : $IMG"

function main {
    echo "./build/main $BATCH $C $W $H $ITER $ACC $OPTI $MODEL $ENGINE $IMG"
    ./build/main $BATCH $C $W $H $ITER $ACC $OPTI $MODEL $ENGINE $IMG
}

if [ $BUILD = "build" ]
then
    echo "Build and Run"

    cd build

    cmake ..

    make all
    cd ..

    sleep 2s
    main
elif [ $BUILD = "run" ]
then
    echo "Run"

    main
fi
