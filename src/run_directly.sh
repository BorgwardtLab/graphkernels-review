#!/bin/bash

if [ -n "$3" ]; then
  poetry run python ../src/train.py ../matrices/$1/$2.npz -n $1 -o ../results/$1_$2.json -I $3
else
  poetry run python ../src/train.py ../matrices/$1/$2.npz -n $1 -o ../results/$1_$2.json
fi
