import logging
from sqlalchemy import create_engine
from config import cli_args, ACCOUNT_BALANCES_DB_URL, BLOCKCHAIN_DB_URL


class BaseApi:
    def __init__(
        self,
        blockchain_db_url: str = BLOCKCHAIN_DB_URL,
        account_balances_db_url: str = ACCOUNT_BALANCES_DB_URL,
    ):
        """
        Creates a new Ethereum protocol API instance
        """
        self.args = cli_args()
        self.logger = logging.getLogger(__name__)
        self.blockchain_db = create_engine(
            blockchain_db_url,
            echo=self.args.verbose,
            connect_args={"check_same_thread": False},
        ).connect()
        self.account_balances_db = create_engine(
            account_balances_db_url,
            echo=self.args.verbose,
            connect_args={"check_same_thread": False},
        ).connect()
