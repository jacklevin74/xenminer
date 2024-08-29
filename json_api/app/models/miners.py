from json_api.app.extensions import db


class Miners(db.Model):
    __tablename__ = "miners"
    __bind_key__ = "difficulty"
    id = db.Column(db.Integer, primary_key=True)
    total_miners = db.Column(db.Integer)
