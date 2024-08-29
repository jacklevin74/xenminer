from json_api.app.extensions import db


class BlockRate(db.Model):
    __tablename__ = "blockrate"
    __bind_key__ = "difficulty"
    id = db.Column(db.Integer, primary_key=True)
    rate = db.Column(db.Float)
