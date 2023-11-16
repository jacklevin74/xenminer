from flask_cors import cross_origin
from api import EthApi
from utils import encode_block
from flask_cors import CORS
import logging
from jsonrpcserver import method, Result, Success, dispatch, InvalidParams, async_dispatch
from flask import Flask, Response, request
import argparse
from web3.types import BlockIdentifier, HexStr
from flask_sock import Sock

BLOCKCHAIN_DB = "sqlite:///blockchain.db"

parser = argparse.ArgumentParser(
    prog="XenBlock RPC/Websocket Server",
    description="Serves RPC and Websocket requests for XenBlock",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--host",
    dest="host",
    action="store",
    default="127.0.0.1",
    help="The host to listen on",
)
parser.add_argument(
    "--port",
    dest="port",
    action="store",
    default=8545,
    help="The port to listen for RPC calls on",
)
parser.add_argument(
    "--verbose", dest="verbose", action="store_true", help="Enable verbose logging"
)
parser.add_argument(
    "--blockchain-db",
    dest="blockchain_db",
    default=BLOCKCHAIN_DB,
    action="store",
    help="Enable debug logging",
)

args = parser.parse_args()

log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(encoding="utf-8", level=log_level)

eth_api = EthApi(args.blockchain_db)
app = Flask(__name__)
sock = Sock(app)
CORS(app)


@app.route("/")
@cross_origin()
def index() -> str:
    return "ok"


@app.route("/", methods=["POST"])
@cross_origin()
def rpc() -> Response:
    return Response(
        dispatch(request.get_data().decode()), content_type="application/json"
    )


@sock.route('/')
@cross_origin()
def echo(ws):
    while True:
        data = ws.receive()
        ws.send(dispatch(data))


@method
def ping() -> Result:
    return Success("pong")


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


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=args.verbose)
