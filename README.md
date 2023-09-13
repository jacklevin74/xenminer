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
