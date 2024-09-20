from json_api.app.extensions import db


class Cache(db.Model):
    __tablename__ = "cache_table"
    __bind_key__ = "cache"
    account = db.Column(db.Text, primary_key=True)
    total_blocks = db.Column(db.Integer)
    hashes_per_second = db.Column(db.Float)
    super_blocks = db.Column(db.Integer)
    rank = db.Column(db.Integer, default=0)
    xnm = db.Column(db.BigInteger, default=0)
    xblk = db.Column(db.BigInteger, default=0)
    xuni = db.Column(db.BigInteger, default=0)
