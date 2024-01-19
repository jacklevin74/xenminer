#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi

find $1 -name "*.py" -exec black {} \;
find $1 -name "*.py" -exec autoflake --in-place --remove-unused-variables {} \;
find $1 -name "*.py" -exec autopep8 --in-place --aggressive --aggressive {} \;
