from decouple import config
import argparse

BLOCKCHAIN_DB_URL = config("BLOCKCHAIN_DB_URL", default="sqlite:///blockchain.db")
ACCOUNT_BALANCES_DB_URL = config(
    "ACCOUNT_BALANCES_DB_URL", default="sqlite:///balances6.db"
)
CHAIN_ID = config("CHAIN_ID", default="100101")
CHAIN_NAME = config("CHAIN_NAME", default="xenblocks")


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
    default=BLOCKCHAIN_DB_URL,
    action="store",
    help="Enable debug logging",
)

args = parser.parse_args()
