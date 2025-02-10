from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class ElevatorDemand(db.Model):
    __tablename__ = 'elevator_demand'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now())
    floor = db.Column(db.Integer, nullable=False)

class ElevatorState(db.Model):
    __tablename__ = 'elevator_state'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now())
    floor = db.Column(db.Integer, nullable=False)
    vacant = db.Column(db.Boolean, nullable=False)