from . import bp
from .service import get_leaderboard, get_leaderboard_entry

from flask import jsonify, request, redirect
from ..extensions import cache


@bp.route("/leaderboard", methods=["GET"])
def redirect_to_explorer():
    return redirect("https://explorer.xenblocks.io")


@bp.route("/v1/leaderboard", methods=["GET"])
@cache.cached(timeout=30, query_string=True)
def index():
    limit = int(request.args.get("limit", 500))
    offset = int(request.args.get("offset", 0))

    return jsonify(get_leaderboard(limit, offset))


@bp.route("/v1/leaderboard/<account>", methods=["GET"])
@cache.cached(timeout=30, query_string=True)
def show(account: str):
    try:
        return jsonify(get_leaderboard_entry(account))
    except ValueError:
        return jsonify({"error": "Account not found"}), 404
