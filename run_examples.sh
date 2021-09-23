#!/bin/bash

for file in pgfexamples/*
do
  echo "Run example ${file}"
  if [[ $file == pgfexamples/skip_*.pgcl ]]
  then
    echo "Skipped!"
  else
    printf "\e[32mResult:\t\e[m"
    time poetry run probably $file --no-simplification 
  fi
done
