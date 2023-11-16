from flask_cors import cross_origin
from eth import EthApi
from eth import encode_block
from net import NetApi
from flask_cors import CORS
from jsonrpcserver import method, Result, Success, dispatch, InvalidParams
from flask import Flask, Response, request
from web3.types import BlockIdentifier, HexStr, Address
from eth_utils import is_hex
from flask_sock import Sock
from config import RPC_MAX_BATCH_SIZE
import logging

eth_api = EthApi()
net_api = NetApi()
app = Flask(__name__)
sock = Sock(app)
CORS(app)

logger = logging.getLogger(__name__)


@app.route("/")
@cross_origin()
def index() -> str:
    return "ok"


@app.route("/", methods=["POST"])
@cross_origin()
def rpc() -> Response:
    data = request.get_data().decode()
    if isinstance(data, list) and len(data) > RPC_MAX_BATCH_SIZE:
        logger.error("batch size too large %d > %d", len(data), RPC_MAX_BATCH_SIZE)
        return Response("Batch size too large", status=400)

    return Response(
        dispatch(data), content_type="application/json"
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


"""
eth methods
"""


@method
def eth_blockNumber() -> Result:
    block = eth_api.block_number()
    return Success(block)


@method
def eth_getBlockByNumber(block_number: BlockIdentifier, full_tx=False) -> Result:
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
def eth_chainId() -> Result:
    return Success(eth_api.chain_id())


@method
def eth_syncing() -> Result:
    return Success(eth_api.syncing())


@method
def eth_gasPrice() -> Result:
    return Success(eth_api.gas_price())


"""
net methods 
"""


@method
def net_version() -> Result:
    return Success(net_api.version())
