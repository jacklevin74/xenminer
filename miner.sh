#!/bin/bash

if [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Error: Windows is not supported yet."
    read
    exit -1
fi

dev_fee_on=false
logging_on=false
opencl=false
silence=false
gpus=0
cpu=false

function display_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -g, --gpus <num>              Set the number of GPUs to use (Default: 1)"
    echo "  -c, --cpu <num>               Running 1 miner in CPU mode (Default: off)"
    echo "  -d, --devfee, --dev-fee-on    Enable dev fee (Default: off)"
    echo "  -o, --opencl                  Enable OpenCL computation (Default: off)"
    echo "  -s, --silence                 Run in silence/background mode (Default: off)"
    echo "  -l, --logging-on              Record verified blocks into payload.log file (Default: off)"
    echo "  -h, --help                    Display this help message and exit"
    echo
    echo "Note: Script exits if both GPU and CPU modes are off."
    exit 0
}
# Process long options
for arg in "$@"; do
    shift
    case "$arg" in
        "--help") set -- "$@" "-h" ;;
        "--opencl") set -- "$@" "-o" ;;
        "--dev-fee-on") set -- "$@" "-d" ;;
        "--devfee") set -- "$@" "-d" ;;
        "--silence") set -- "$@" "-s" ;;
        "--gpus") set -- "$@" "-g" ;;
        "--cpu") set -- "$@" "-c" ;;
        "--logging-on") set -- "$@" "-l" ;;
         *) set -- "$@" "$arg"
    esac
done

# Now, process short options with getopts
while getopts "g:c:ldosh" opt; do
    case "$opt" in
    g) gpus="$OPTARG" ;;
    c) cpu="$OPTARG" ;;
    d) dev_fee_on=true ;;
    o) opencl=true ;;
    s) silence=true ;;
    l) logging_on=true ;;
    h) display_help ;;
    *) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# cpu mode
if [ $cpu -gt 0 ]; then
    echo "Running $cpu miner in CPU mode..."

    command="./xengpuminer -m cpu"
    for ((i = 0; i < $cpu; i++)); do
        if [ $i -eq 0 ]; then
            screen -S "cpuminer" -dm bash -c "$command"
        else
            screen -S "cpuminer" -X screen bash -c "$command"
        fi
    done
fi

# gpu mode
if [[ $gpus -lt 1 && $cpu -lt 1 ]]; then
    echo "Error: Neither gpu nor cpu mode is selected."
    exit -1
fi
for ((i = 0; i < $gpus; i++)); do
    command="./xengpuminer -d $i"
    if $opencl; then
        command+=" -m opencl"
    fi
    if [ $i -eq 0 ]; then
        screen -S "gpuminer" -dm bash -c "$command"
    else
        screen -S "gpuminer" -X screen bash -c "$command"
    fi
done
if [ $gpus -gt 0 ]; then
    echo "Running $gpus miners in GPU mode..."
fi
command="python3 miner.py"
if $dev_fee_on; then
    command+=" --dev-fee-on"
fi
if $logging_on; then
    command+=" --logging-on"
fi
if $silence; then
    screen -S submitminer -dm bash -c "$command"
    echo "If you want to stop, run: pkill xengpuminer && pkill -f submitminer. or simply use pkill -f miner"
else
    bash -c "$command"
    pkill -f "submitminer"
    pkill gpuminer
fi
