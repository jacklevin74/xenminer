import argparse
import logging
from decouple import config

BLOCKCHAIN_DB_URL = config(
    "BLOCKCHAIN_DB_URL",
    default="sqlite:///blockchain.db")
ACCOUNT_BALANCES_DB_URL = config(
    "ACCOUNT_BALANCES_DB_URL", default="sqlite:///balances6.db"
)
CHAIN_ID = config("CHAIN_ID", default="100101")
CHAIN_NAME = config("CHAIN_NAME", default="xenblocks")
RPC_MAX_BATCH_SIZE = config("RPC_MAX_BATCH_SIZE", default=10)
RPC_PORT = config("RPC_PORT", default=8545)
RPC_HOST = config("RPC_HOST", default="0.0.0.0")


def cli_args():
    parser = argparse.ArgumentParser(
        prog="Xenium RPC/Websocket Server",
        description="Serves RPC and Websocket requests for Xenium",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        dest="host",
        action="store",
        default=RPC_HOST,
        help="The host to listen on",
    )
    parser.add_argument(
        "--port",
        dest="port",
        action="store",
        default=RPC_PORT,
        help="The port to listen for RPC calls on",
    )
    parser.add_argument(
        "--dev",
        dest="dev",
        action="store_true",
        help="Use the development server instead of gunicorn",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose logging")

    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning("Unknown args: %s", unknown)

    return args
