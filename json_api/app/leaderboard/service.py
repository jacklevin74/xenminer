from sqlalchemy import func
from ..models.blockrate import BlockRate
from ..models.blocks import Blocks
from ..models.cache import Cache
from ..models.miners import Miners
from gpage import get_difficulty
from eth_utils import to_checksum_address


def _fmt(value, as_string):
    return str(value) if as_string else value


def fetch_cache_data(limit, offset, require_sol_address):
    query = Cache.query.order_by(Cache.total_blocks.desc())
    if require_sol_address:
        query = query.filter(Cache.sol_address.isnot(None))
    return query.limit(limit).offset(offset).all()


def fetch_latest_rate():
    return BlockRate.query.order_by(BlockRate.id.desc()).first()


def fetch_latest_miners():
    return Miners.query.order_by(Miners.id.desc()).first()


def fetch_total_blocks():
    return Blocks.query.order_by(Blocks.block_id.desc()).first()


def fetch_token_totals(as_string=False):
    result = Cache.query.with_entities(
        func.sum(Cache.xnm).label("total_xnm"),
        func.sum(Cache.xuni).label("total_xuni"),
        func.sum(Cache.xblk).label("total_xblk"),
    ).first()
    return {
        "totalXnm": _fmt(result.total_xnm or 0, as_string),
        "totalXuni": _fmt(result.total_xuni or 0, as_string),
        "totalXblk": _fmt(result.total_xblk or 0, as_string),
    }


def fetch_sol_token_totals(as_string=False):
    result = Cache.query.with_entities(
        func.sum(Cache.xnm).label("total_xnm"),
        func.sum(Cache.xuni).label("total_xuni"),
        func.sum(Cache.xblk).label("total_xblk"),
    ).filter(Cache.sol_address.isnot(None)).first()
    return {
        "totalXnmWithSol": _fmt(result.total_xnm or 0, as_string),
        "totalXuniWithSol": _fmt(result.total_xuni or 0, as_string),
        "totalXblkWithSol": _fmt(result.total_xblk or 0, as_string),
    }


def get_leaderboard(limit: int, offset: int, require_sol_address: bool = False,
                    string_values: bool = False):
    difficulty = get_difficulty()
    cache_data = fetch_cache_data(limit, offset, require_sol_address)
    latest_rate = fetch_latest_rate()
    latest_miners = fetch_latest_miners()
    total_blocks = fetch_total_blocks()
    token_totals = fetch_token_totals(as_string=string_values)
    sol_token_totals = fetch_sol_token_totals(as_string=string_values)

    latest_rate = latest_rate.rate if latest_rate else 0
    latest_miners = latest_miners.total_miners if latest_miners else 0
    total_blocks = total_blocks.block_id if total_blocks else None

    miners = [
        {
            "rank": r.rank,
            "account": to_checksum_address(r.account.strip()),
            "blocks": r.total_blocks,
            "hashRate": round(r.hashes_per_second, 2),
            "superBlocks": r.super_blocks,
            "xnm": _fmt(r.xnm, string_values),
            "xblk": _fmt(r.xblk, string_values),
            "xuni": _fmt(r.xuni, string_values),
            "solAddress": r.sol_address.strip() if r.sol_address else None,
        }
        for i, r in enumerate(cache_data)
    ]

    return {
        "totalHashRate": latest_rate,
        "totalMiners": latest_miners,
        "totalBlocks": total_blocks,
        "difficulty": difficulty,
        **token_totals,
        **sol_token_totals,
        "miners": miners,
    }


def get_leaderboard_entry(account: str, string_values: bool = False):
    result = Cache.query.filter_by(account=account.lower()).first()
    if not result:
        raise ValueError("Account not found")

    return {
        "account": to_checksum_address(result.account.strip()),
        "blocks": result.total_blocks,
        "hashRate": round(result.hashes_per_second, 2),
        "superBlocks": result.super_blocks,
        "rank": result.rank,
        "xnm": _fmt(result.xnm, string_values),
        "xblk": _fmt(result.xblk, string_values),
        "xuni": _fmt(result.xuni, string_values),
        "solAddress": result.sol_address.strip() if result.sol_address else None,
    }

