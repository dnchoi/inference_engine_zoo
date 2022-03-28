#!/bin/sh

# commend

# exit on first error
set -e

BUILD=$1
INFERENCE=$2

echo "BUILD or RUN : $BUILD"

function main {
    ./build/main 
}

if [ $BUILD = "build" ]
then
    echo "Build and Run"

    cd build
    cmake -DWITH_INFERENCE_ENGINE=${INFERENCE} ..
    make -j20
    cd ..

    main
elif [ $BUILD = "run" ]
then
    echo "Run"

    main
fi
