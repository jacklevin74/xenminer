from web3.types import BlockData


def encode_block(block: BlockData) -> dict[str, str]:
    """
    Encodes the web3 block object into an Etheruem RPC Block response dictionary
    which is ready to be serialized into JSON
    """

    eb = block.copy()

    # HexBytes to hex string
    for key in [
        "hash",
        "parentHash",
        "nonce",
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
            eb[key] = "0x0"  # type: ignore

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
            eb[key] = "0x0"  # type: ignore

    eb["transactions"] = []
    eb["uncles"] = []

    return eb
