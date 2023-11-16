from hexbytes import HexBytes
from web3.types import BlockNumber, Timestamp, BlockData, BlockIdentifier, Address
from sqlalchemy import text
from eth_utils import is_hex
from base import BaseApi
from config import CHAIN_ID


class EthApi(BaseApi):
    def block_number(self) -> BlockNumber:
        """
        Returns the current block number
        """
        self.logger.debug("block_number()")
        res = self.blockchain_db.execute(
            text(
                """
                SELECT id - 1
                FROM blockchain
                ORDER BY id DESC
                LIMIT 1
                """
            )
        )
        row = res.fetchone()
        if not row:
            return BlockNumber(0)
        return BlockNumber(row[0])

    def get_block_by_hash(
        self, block_hash: BlockIdentifier, full_tx: bool = True
    ) -> BlockData | None:
        """
        Returns information of the block matching the given block hash
        """
        self.logger.debug("block_by_hash(%s, %s)", block_hash, full_tx)
        if not block_hash:
            return None

        block_hash = block_hash.replace("0x", "")
        res = self.blockchain_db.execute(
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

        self.logger.debug("found blockchain record %s", row)

        b = BlockData()

        try:
            b["number"] = BlockNumber(row[0])
            b["timestamp"] = Timestamp(int(row[1]))
            b["hash"] = HexBytes.fromhex(row[2])

            if row[3] != "genesis":
                b["parentHash"] = HexBytes.fromhex(row[3])
        except RuntimeError as e:
            self.logger.error("failed to read block data", {"row": row})
            raise e
        return b

    def eth_get_block_by_number(
        self, block_number: BlockIdentifier, full_tx: bool = False
    ) -> BlockData | None:
        """
        Returns information of the block matching the given block number.
        """
        self.logger.debug("block_by_number(%s, %s)", block_number, full_tx)
        if isinstance(block_number, str) and is_hex(block_number):
            block_number = int(block_number, 16)

        return self.get_block_by_hash(
            self._get_block_hash_by_number(block_number), full_tx
        )

    def _get_block_hash_by_number(
        self, block_number: BlockIdentifier
    ) -> BlockIdentifier | None:
        self.logger.debug("_get_block_hash_by_number(%s)", block_number)

        if block_number == "earliest":
            block_number = 0

        if block_number == "latest":
            res = self.blockchain_db.execute(
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
            res = self.blockchain_db.execute(
                text(
                    """
                    SELECT block_hash
                    FROM blockchain
                    WHERE id = :block_number
                    """
                ),
                {"block_number": block_number + 1},
            )
        row = res.fetchone()
        if not row:
            return None
        return row[0]

    def syncing(self) -> bool:
        """
        Returns true if the node is syncing
        """
        # TODO: implement
        return False

    def chain_id(self) -> int:
        """
        Returns the chain ID
        """
        return CHAIN_ID

    def gas_price(self) -> int:
        """
        Returns the current gas price
        """
        # TODO: implement
        return 0

    def get_balance(self, address: Address, block_number: BlockIdentifier) -> HexBytes:
        """
        Returns the balance of the given address
        """
        self.logger.debug("eth_getBalance(%s, %s)", address, block_number)

        if block_number == "latest":
            block_number = self.block_number()
        elif block_number == "earliest":
            block_number = 0
        elif isinstance(block_number, str) and is_hex(block_number):
            block_number = int(block_number, 16)

        row = self.account_balances_db.execute(
            text(
                """
                SELECT sum(amount * 10)
                FROM account_balances
                WHERE account = :address
                    AND block_id <= :block_number
                    AND currency_type = 1
                """
            ),
            {"address": address, "block_number": block_number},
        ).fetchone()

        print(row)

        if not row:
            return HexBytes("0x0")
        return HexBytes(hex(row[0]))


def encode_block(block: BlockData) -> dict[str, str]:
    """
    Encodes the web3 block object into an Etheruem RPC Block response dictionary
    which is ready to be serialized into JSON
    """

    eb = block.copy()

    if "parentHash" in block:
        eb["parentHash"] = block["parentHash"].hex()
    else:
        eb[
            "parentHash"
        ] = "0x0000000000000000000000000000000000000000000000000000000000000000"

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
