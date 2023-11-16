## Overview

Introduction:  

This proof of work miner is based on Argon2ID algorithm which is both GPU and ASIC resistant.
It allows all participants to mine blocks fairly.  Your mining speed is directly proportional to 
the number of miners you are running (you can run many on a single computer).  The difficulty of 
mining is auto adjusted based on the verifier node algorithm which aproximately targets production
speed of 1 block per second.

## Installation

Install all the required modules by executing the command below.  Make sure you have at least python3 and pip3 installed in order to proceed.

```bash
pip install -U -r requirements.txt
```
## 

To start your miner just execute this command.  Note you should adjust account at the top of the file to be your ethereum address if you want to claim your blocks and superblocks later

```bash
python3 miner.py
```

## Ethereum Compatible RPC/Websocket API

### Support Methods

- net_version
- eth_chainId
- eth_blockNumber
- eth_getBlockByNumber
- eth_getBlockByHash
- eth_getBalance

### Usage

> Run the server
```shell
python3 ethapi
```

#### Examples

> Get the latest block number
```shell
curl -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' http://localhost:8545
```

> Get the latest Block using websockets ([wscat](https://github.com/websockets/wscat))
```shell
wscat -c ws://localhost:8545
> {"jsonrpc":"2.0","method":"eth_getBlockByNumber","params":["latest", true],"id":2}
< {"jsonrpc": "2.0", "result": {"number": "0x1fdd4", "timestamp": "0x65565b2b", "hash": "0xf93fbcb8b41de542b2509c897a86a4a692c04fe078880f96cad8f47c35e818d7", "parentHash": "0xcdbfadb7c01000d8ed3811e8f2e45f29156baab21adc66c5d8ae6a0d82634a25", "sha3Uncles": "0x0000000000000000000000000000000000000000", "logsBloom": "0x0000000000000000000000000000000000000000", "transactionsRoot": "0x0000000000000000000000000000000000000000", "stateRoot": "0x0000000000000000000000000000000000000000", "receiptsRoot": "0x0000000000000000000000000000000000000000", "miner": "0x0000000000000000000000000000000000000000", "extraData": "0x0000000000000000000000000000000000000000", "difficulty": "0x0000000000000000000000000000000000000000", "totalDifficulty": "0x0000000000000000000000000000000000000000", "size": "0x0000000000000000000000000000000000000000", "gasLimit": "0x0000000000000000000000000000000000000000", "gasUsed": "0x0000000000000000000000000000000000000000", "nonce": "0x0000000000000000", "transactions": [], "uncles": []}, "id": 2}
```
