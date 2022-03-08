#!/bin/sh

# commend
# bash build_run.sh run 0 img
# bash build_run.sh run 0 img.png
# bash build_run.sh run 1000 img.png

# bash build_run.sh build 0 img
# bash build_run.sh build 0 img.png
# bash build_run.sh build 1000 img.png


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
    ./build/main $BATCH $C $W $H $ITER $ACC $OPTI $MODEL $ENGINE $IMG
}

if [ $BUILD = "build" ]
then
    echo "Build and Run"

    cd build

    # Generate a Makefile for GCC (or Clang, depanding on CC/CXX envvar)
    cmake ..

    # Build (ie 'make')
    # cmake --build .
    make all
    cd ..

    sleep 2s
    echo "One cycle"
    main
elif [ $BUILD = "run" ]
then
    echo "Run"

    echo "One cycle"
    main
fi

# args._B = atoi(argv[1]);
# args._W = atoi(argv[2]);
# args._H = atoi(argv[3]);
# args._C = atoi(argv[4]);
# args._iter = atoi(argv[5]);
# args._acc = atoi(argv[6]);
# args._opti = atoi(argv[6]);
# args._model = argv[7];
# args._engine = argv[8];

