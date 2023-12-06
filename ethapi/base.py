import logging
from ethapi.config import cli_args
from db.blockchain import connect as blockchain_connect
from db.account_balances import connect as account_balances_connect


class BaseApi:
    def __init__(self):
        """
        Creates a new Ethereum protocol API instance
        """
        self.args = cli_args()
        self.logger = logging.getLogger(__name__)
        self.blockchain_db = blockchain_connect()
        self.account_balances_db = account_balances_connect()
