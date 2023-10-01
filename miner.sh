#!/bin/bash
dev_fee_on=false
opencl=false
runs=1
while getopts "c:do" opt; do
    case "$opt" in
    c) runs="$OPTARG" ;;
    d) dev_fee_on=true ;;
    o) opencl=true ;;
    esac
done

command="python3 miner.py"
if $dev_fee_on; then
    command+=" --dev-fee-on"
fi

screen -S "miner" -dm bash -c "$command"

screen -S "gpuminer" -dm
for ((i = 0; i < $runs; i++)); do
    command="./xengpuminer -d $i"
    if $opencl; then
        command+=" -m opencl"
    fi
    screen -S "gpuminer" -X screen bash -c "$command"
done

echo "Successfully started $runs processes."