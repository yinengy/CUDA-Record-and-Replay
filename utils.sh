#!/bin/bash

# change it to the corresponding path
app_dir='test-apps'
tool_dir='tools'

# help
usage() {
    echo "------------------For Application------------------"
    echo "Source and makefile for application should to be put into <app_dir>/<app name>/"
    echo "The compiled binary should have the same name with the directory it is in."
    echo "$0 make <app name> : call the makefile of the application."
    echo ""
    echo "------------------For NVBit Tool-------------------"
    echo "Source and makefile for NVBit tool should to be put into <tool_dir>/<tool name>/"
    echo "The compiled binary should have the same name (with a .so extension) with the directory it is in."
    echo "$0 make_tool <app name> : call the makefile of the NVbit tool."
    echo ""
    echo "------------------Execution------------------------"
    echo "Tool and application will be compiled if not."
    echo "$0 <tool name> <app name> : run <app name> with <tool name>"
}

case $1 in
  "help")
    usage
    ;;

  "make")
    make --directory $app_dir/$2
    ;;

  "make_tool")
    make --directory $tool_dir/$2
    ;;

  "race_check")
    chmod +x scripts/race_check_helper.py
    # will combine with scripts/print_data_race_helper.py
    make --directory $tool_dir/$1

    make --directory $app_dir/$2

    LD_PRELOAD=$tool_dir/$1/$1.so ./$app_dir/$2/run | scripts/race_check_helper.py
    ;;
  
  "race_check_trace")
    chmod +x scripts/race_check_helper.py
    # will combine with scripts/print_data_race_helper.py
    make --directory $tool_dir/$1

    make --directory $app_dir/$2

    LD_PRELOAD=$tool_dir/$1/$1.so ./$app_dir/$2/run | scripts/race_check_helper.py
    ;;

  *)
    make --directory $tool_dir/$1

    make --directory $app_dir/$2

    LD_PRELOAD=$tool_dir/$1/$1.so ./$app_dir/$2/run
    ;;
esac