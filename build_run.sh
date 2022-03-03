#!/bin/sh

# exit on first error
set -e
rm -rf build
mkdir -p build
cd build

# Generate a Makefile for GCC (or Clang, depanding on CC/CXX envvar)
cmake ..

# Build (ie 'make')
# cmake --build .
make all
cd ..

./build/main
