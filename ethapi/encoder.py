from web3.types import BlockData, TxData, TxReceipt


def encode_block(block: BlockData) -> dict[str, str] | None:
    """
    Encodes the web3 block object into an Ethereum RPC Block response dictionary
    which is ready to be serialized into JSON
    """

    if not block:
        return None

    eb = block.copy()

    if "parentHash" in block:
        eb["parentHash"] = block["parentHash"].hex()
    else:
        eb[
            "parentHash"
        ] = "0x0000000000000000000000000000000000000000000000000000000000000000"

    # HexBytes to hex string
    for key in [
        "hash",
        "sha3Uncles",
        "logsBloom",
        "transactionsRoot",
        "stateRoot",
        "receiptsRoot",
        "miner",
        "extraData",
    ]:
        if key in block:
            eb[key] = block[key].hex()  # type: ignore
        else:
            eb[key] = "0x0000000000000000000000000000000000000000"  # type: ignore

    # ints to hex strings
    for key in [
        "number",
        "difficulty",
        "totalDifficulty",
        "size",
        "gasLimit",
        "gasUsed",
        "timestamp",
    ]:
        if key in block:
            eb[key] = hex(block[key])  # type: ignore
        else:
            eb[key] = "0x0000000000000000000000000000000000000000"  # type: ignore

    eb["nonce"] = "0x0000000000000000"
    eb["transactions"] = []
    eb["uncles"] = []

    return eb


def encode_transaction(transaction: TxData) -> dict[str, str] | None:
    """
    Encodes the web3 transaction object into an Ethereum RPC transaction response dictionary
    which is ready to be serialized into JSON
    """

    if not transaction:
        return None

    et = transaction.copy()

    # HexBytes to hex string
    for key in ["blockHash", "from", "to"]:
        if key in transaction:
            et[key] = transaction[key].hex()  # type: ignore
        else:
            et[key] = "0x0000000000000000000000000000000000000000"  # type: ignore

    # ints to hex strings
    for key in [
        "blockNumber",
        "gas",
        "gasPrice",
        "nonce",
        "transactionIndex",
        "value",
        "v",
        "r",
        "s",
    ]:
        if key in transaction:
            et[key] = hex(transaction[key])  # type: ignore
        else:
            et[key] = "0x0000000000000000000000000000000000000000"  # type: ignore

    return et


def encode_transaction_receipt(
        transaction_receipt: TxReceipt) -> dict[str, str] | None:
    """
    Encodes the web3 transaction receipt object into an Ethereum RPC transaction receipt response dictionary
    which is ready to be serialized into JSON
    """

    if not transaction_receipt:
        return None

    etr = transaction_receipt.copy()

    # HexBytes to hex string
    for key in ["blockHash", "from", "to", "contractAddress", "logsBloom"]:
        if key in transaction_receipt:
            etr[key] = transaction_receipt[key].hex()  # type: ignore
        else:
            etr[key] = "0x0000000000000000000000000000000000000000"  # type: ignore

    # ints to hex strings
    for key in [
        "blockNumber",
        "cumulativeGasUsed",
        "effectiveGasPrice",
        "gasUsed",
        "nonce",
        "transactionIndex",
        "value",
    ]:
        if key in transaction_receipt:
            etr[key] = hex(transaction_receipt[key])  # type: ignore
        else:
            etr[key] = "0x0000000000000000000000000000000000000000"  # type: ignore

    etr["logs"] = []
    etr["type"] = "0x1"
    etr["status"] = "0x1"

    return etr
