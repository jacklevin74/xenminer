from . import bp


@bp.route("/health")
def index():
    return "ok"
