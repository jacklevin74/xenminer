#!/bin/bash

if [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Error: Windows is not supported yet."
    read
    exit -1
fi

dev_fee_on=false
opencl=false
silence=false
gpus=1
cpucores=0

function display_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -g, --gpus <num>              Set the number of GPUs to use (Default: 1)"
    echo "  -c, --cpucores <num>          Set the number of CPU cores to use; Only CPU mode is activated when >0 (Default: 0)"
    echo "  -d, --devfee, --dev-fee-on    Enable dev fee (Default: off)"
    echo "  -o, --opencl                  Enable OpenCL computation (Default: off)"
    echo "  -s, --silence                 Run in silence/background mode (Default: off)"
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
        "--cpucores") set -- "$@" "-c" ;;
         *) set -- "$@" "$arg"
    esac
done

# Now, process short options with getopts
while getopts "g:c:dosh" opt; do
    case "$opt" in
    g) gpus="$OPTARG" ;;
    c) cpucores="$OPTARG" ;;
    d) dev_fee_on=true ;;
    o) opencl=true ;;
    s) silence=true ;;
    h) display_help ;;
    *) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# cpu mode
if [ $cpucores -gt 0 ]; then
    command="python3 miner.py --gpu=false"
    if $dev_fee_on; then
        command+=" --dev-fee-on"
    fi
    echo "Running in CPU mode with $cpucores cores."
    if [ $cpucores -gt 99 ]; then
        echo "CPU cores must be less than 100. More is supported later."
        exit -1
    fi
    if $silence; then
        for ((i = 0; i < $cpucores; i++)); do
            if [ $i -eq 0 ]; then
                screen -S "miner" -dm bash -c "$command"
            else
                screen -S "miner" -X screen bash -c "$command"
            fi
        done    
    else
        echo "Running 1 miner in CPU mode..."
        bash -c "$command"
    fi
    exit 0
fi

# gpu mode
if [ $gpus -lt 1 ]; then
    echo "Error: GPU number must be greater than 0 or run in CPU mode."
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

command="python3 miner.py"
if $dev_fee_on; then
    command+=" --dev-fee-on"
fi
echo "Successfully started $gpus gpuminers"
if $silence; then
    screen -S "miner" -dm bash -c "$command"
    echo "If you want to stop, run: pkill screen"
else
    bash -c "$command"
    bash -c "pkill screen"
fi
