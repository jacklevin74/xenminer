from sqlalchemy import text
from sqlalchemy import create_engine
import logging
import json
from ethapi.types import BlockData, TransactionData
import hashlib
DEFAULT_DB_URL = "sqlite+pysqlite:///:memory:"
DEFAULT_LOG_LEVEL = logging.INFO


class EthApi:
    def __init__(self, db_url: str = DEFAULT_DB_URL):
        """Creates a new Ethereum protocol API instance"""
        self.logger = logging.getLogger(__name__)
        self.engine = create_engine(
            db_url, echo=True, connect_args={"check_same_thread": False}
        )
        self.conn = self.engine.connect()

    def block_by_hash(self, block_hash, full_tx: bool = True) -> BlockData | None:
        """
        GetBlockByHash returns the requested block. When fullTx is true all transactions in the block are returned in full
        detail, otherwise only the transaction hash is returned.
        """
        logging.debug("block_by_hash")
        if not block_hash:
            return None

        block_hash = block_hash.replace("0x", "")
        res = self.conn.execute(
            text(
                """
                SELECT id - 1,
                       strftime('%s', timestamp),
                       block_hash,
                       prev_hash,
                       records_json
                FROM blockchain
                WHERE block_hash = :block_hash
                """
            ),
            {"block_hash": block_hash},
        )

        row = res.fetchone()
        if not row:
            return None

        b = BlockData()
        b.number = hex(row[0])
        b.timestamp = hex(int(row[1]))
        b.hash = row[2]
        b.parentHash = row[3]

        transactions = json.loads(row[4])
        for idx, tx in enumerate(transactions):

            if "id" in tx:
                m = hashlib.sha256(f'{tx["account"]}{tx["id"]}10'.encode())
            else:
                m = hashlib.sha256(f'{tx["account"]}{tx["block_id"]}10'.encode())

            tx_hash = m.hexdigest()

            if full_tx:
                tx_obj = TransactionData()
                tx_obj.blockHash = b.hash
                tx_obj.from_ = tx["account"]
                tx_obj.to = "0x0"
                tx_obj.gas = 0
                tx_obj.gasPrice = 0
                tx_obj.hash = tx_hash
                tx_obj.transactionIndex = hex(idx)
                b.transactions.append(tx_obj)
            else:
                b.transactions.append(tx_hash)
        return b

    def block_by_number(
        self, block_number: str = "latest", full_tx: bool = False
    ) -> BlockData | None:
        logging.debug("block_by_number", {block_number, full_tx})
        return self.block_by_hash(self._get_block_hash_by_number(block_number), full_tx)

    def _get_block_hash_by_number(self, block_number):
        logging.debug("_get_block_hash_by_number")

        if block_number == "earliest":
            block_number = 1

        if block_number == "latest":
            res = self.conn.execute(
                text(
                    """
                    SELECT block_hash
                    FROM blockchain
                    ORDER BY id DESC
                    LIMIT 1
                    """
                )
            )
        else:
            res = self.conn.execute(
                text(
                    """
                    SELECT block_hash
                    FROM blockchain
                    WHERE id = :block_number
                    """
                ),
                {"block_number": block_number},
            )
        row = res.fetchone()
        if not row:
            return None
        return row[0]
