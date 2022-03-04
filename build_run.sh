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

input=$1
input2=$2
input3=$3
echo $input
echo $input2
echo $input3

function main {
    ./build/main 1 3 112 112 1000 1 0 model.onnx onnx $input3
}

if [ $input = "build" ]
then
    echo "Build and Run"
    rm -rf build
    sleep 2s
    mkdir build
    cd build

    # Generate a Makefile for GCC (or Clang, depanding on CC/CXX envvar)
    cmake ..

    # Build (ie 'make')
    # cmake --build .
    make all
    cd ..
    if [ $input2 != 0 ]
    then
        for ((i=0;i<=$input2;i++))
        do
            main
            echo "Running loop "$i

        done
    else
        echo "One cycle"
        main
    fi
elif [ $input = "run" ]
then
    echo "Run"
    if [ $input2 != 0 ]
    then
        for ((i=0;i<=$input2;i++))
        do
            main
            echo "Running loop "$i

        done
    else
        echo "One cycle"
        main
    fi
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

