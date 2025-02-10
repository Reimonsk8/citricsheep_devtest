from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from app.routes import api
from utils.random_data_generator import insert_elevator_data_into_db, delete_all_data_from_tables

# due there is no training data for the model activate this option to generate it randomly
OPTIONS = {
    "generate_random_test_data": True
}
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///elevator.db'
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    # api routes Register the blueprint appis
    app.register_blueprint(api)
    #create database tables
    with app.app_context():
        db.create_all()
        # enable random data generation and insertion into the database
        if OPTIONS["generate_random_test_data"]:
            delete_all_data_from_tables()
            insert_elevator_data_into_db()

    return app

app = create_app()
if __name__ == "__main__":
    app.run(debug=True) 
