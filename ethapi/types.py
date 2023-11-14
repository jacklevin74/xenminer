class BlockData:
    def __init__(self):
        self.number = "0x0"
        self.baseFeePerGas = "0x0"
        self.difficulty = "0x0"
        self.extraData = "0x0"
        self.gasLimit = "0x0"
        self.gasUsed = "0x0"
        self.hash = "0x0"
        self.logsBloom = "0x0"
        self.miner = "0x0"
        self.mixHash = "0x0"
        self.nonce = "0x0"
        self.parentHash = "0x0"
        self.receiptsRoot = "0x0"
        self.sha3Uncles = "0x0"
        self.size = "0x0"
        self.stateRoot = "0x0"
        self.timestamp = "0x0"
        self.totalDifficulty = "0x0"
        self.transactions = []
        self.transactionsRoot = "0x0"
        self.uncles = []


class TransactionData:
    def __init__(self):
        self.blockHash = "0x0"
        self.blockNumber = "0x0"
        self.from_ = "0x0"
        self.gas = "0x0"
        self.gasPrice = "0x0"
        self.hash = "0x0"
        self.input = "0x0"
        self.nonce = "0x0"
        self.r = "0x0"
        self.s = "0x0"
        self.to = "0x0"
        self.transactionIndex = "0x0"
        self.v = "0x0"
        self.value = "0x0"
