from ..models.blockrate import BlockRate
from ..models.blocks import Blocks
from ..models.cache import Cache
from ..models.miners import Miners
from gpage import get_difficulty


def fetch_cache_data(limit, offset):
    return (
        Cache.query.order_by(Cache.total_blocks.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )


def fetch_latest_rate():
    return BlockRate.query.order_by(BlockRate.id.desc()).first()


def fetch_latest_miners():
    return Miners.query.order_by(Miners.id.desc()).first()


def fetch_total_blocks():
    return Blocks.query.order_by(Blocks.block_id.desc()).first()


def get_leaderboard(limit: int, offset: int):
    difficulty = get_difficulty()
    cache_data = fetch_cache_data(limit, offset)
    latest_rate = fetch_latest_rate()
    latest_miners = fetch_latest_miners()
    total_blocks = fetch_total_blocks()

    latest_rate = latest_rate.rate if latest_rate else 0
    latest_miners = latest_miners.total_miners if latest_miners else 0
    total_blocks = total_blocks.block_id if total_blocks else None

    miners = [
        {
            "rank": r.rank,
            "account": r.account.strip(),
            "blocks": r.total_blocks,
            "hashRate": round(r.hashes_per_second, 2),
            "superBlocks": r.super_blocks,
        }
        for i, r in enumerate(cache_data)
    ]

    return {
        "totalHashRate": latest_rate,
        "totalMiners": latest_miners,
        "totalBlocks": total_blocks,
        "difficulty": difficulty,
        "miners": miners,
    }


def get_leaderboard_entry(account: str):
    result = Cache.query.filter_by(account=account.lower()).first()
    if not result:
        raise ValueError("Account not found")

    return {
        "account": result.account.strip(),
        "blocks": result.total_blocks,
        "hashRate": round(result.hashes_per_second, 2),
        "superBlocks": result.super_blocks,
        "rank": result.rank,
    }
