import logging
from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
from flask_sock import Sock
from jsonrpcserver import method, Result, Success, dispatch, InvalidParams, JsonRpcError
from jsonrpcserver.main import default_validator
from web3.types import BlockIdentifier, HexStr, Address
from eth_utils import is_hex

from ethapi.encoder import encode_block, encode_transaction, encode_transaction_receipt
from ethapi.eth import EthApi
from ethapi.net import NetApi
from ethapi.config import RPC_MAX_BATCH_SIZE


eth_api = EthApi()
net_api = NetApi()
app = Flask(__name__)
sock = Sock(app)
CORS(app)

logger = logging.getLogger(__name__)


def _validator(data):
    """
    Validate the request data
    """
    if isinstance(data, list) and len(data) > RPC_MAX_BATCH_SIZE:
        logger.error(
            "batch size too large %s > %s",
            len(data),
            RPC_MAX_BATCH_SIZE)
        raise JsonRpcError(400, "Batch size too large")
    return default_validator(data)


@app.route("/")
@cross_origin()
def index() -> str:
    return "ok"


@app.route("/", methods=["POST"])
@cross_origin()
def rpc() -> Response:
    data = request.get_data().decode()
    logger.debug("rpc request: %s", data)
    return Response(
        dispatch(data, validator=_validator), content_type="application/json"
    )


@sock.route("/")
@cross_origin()
def echo(ws):
    while True:
        data = ws.receive()
        ws.send(dispatch(data))


@method
def ping() -> Result:
    return Success("pong")


###############
# eth methods #
###############


@method
def eth_blockNumber() -> Result:
    block = eth_api.block_number()
    return Success(block)


@method
def eth_getBlockByNumber(
        block_number: BlockIdentifier,
        full_tx=False) -> Result:
    block = eth_api.eth_get_block_by_number(block_number, full_tx)
    return Success(encode_block(block))


@method
def eth_getBlockByHash(block_hash: HexStr, full_tx=False) -> Result:
    if not isinstance(block_hash, str):
        return InvalidParams("block_hash must be a hex string")
    block = eth_api.get_block_by_hash(block_hash, full_tx)
    return Success(encode_block(block))


@method
def eth_getBalance(address: str, block_number: BlockIdentifier) -> Result:
    if not isinstance(address, str) or not is_hex(address):
        return InvalidParams("address must be a hex string")

    address_bytes = Address(bytes.fromhex(address.replace("0x", "")))
    balance = eth_api.get_balance(address_bytes, block_number)
    return Success(balance.hex())


@method
def eth_getTransactionByHash(tx_hash: HexStr) -> Result:
    if not isinstance(tx_hash, str):
        return InvalidParams("tx_hash must be a hex string")
    tx = eth_api.get_transaction_by_hash(tx_hash)
    return Success(encode_transaction(tx))


@method
def eth_getTransactionByBlockNumberAndIndex(
    block_number: BlockIdentifier, tx_index: int
) -> Result:
    tx = eth_api.get_transaction_by_block_number_and_index(
        block_number, tx_index)
    return Success(encode_transaction(tx))


@method
def eth_getTransactionByBlockHashAndIndex(
        block_hash: HexStr,
        tx_index: int) -> Result:
    if not isinstance(block_hash, str):
        return InvalidParams("block_hash must be a hex string")

    tx = eth_api.get_transaction_by_block_hash_and_index(block_hash, tx_index)
    return Success(encode_transaction(tx))


@method
def eth_getTransactionReceipt(tx_hash: HexStr) -> Result:
    if not isinstance(tx_hash, str):
        return InvalidParams("tx_hash must be a hex string")
    tx_rec = eth_api.get_transaction_receipt(tx_hash)
    return Success(encode_transaction_receipt(tx_rec))


@method
def eth_chainId() -> Result:
    return Success(eth_api.chain_id())


@method
def eth_syncing() -> Result:
    return Success(eth_api.syncing())


@method
def eth_gasPrice() -> Result:
    return Success(eth_api.gas_price())


###############
# net methods #
###############


@method
def net_version() -> Result:
    return Success(net_api.version())
