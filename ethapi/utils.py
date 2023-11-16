from web3.types import BlockData


def encode_block(block: BlockData) -> dict[str, str]:
    """
    Encodes the web3 block object into an Etheruem RPC Block response dictionary
    which is ready to be serialized into JSON
    """

    eb = block.copy()

    if "parentHash" in block:
        eb["parentHash"] = block["parentHash"].hex()
    else:
        eb["parentHash"] = "0x0000000000000000000000000000000000000000000000000000000000000000"

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
