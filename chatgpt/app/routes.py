from flask import Blueprint, request, jsonify
from app.models import db, ElevatorDemand, ElevatorState  # Import models and db

api = Blueprint('api', __name__)

@api.route('/api/demand', methods=['POST'])
def create_demand():
    try:
        data = request.get_json()
        new_demand = ElevatorDemand(floor=data['floor'])
        db.session.add(new_demand)
        db.session.commit()
        return jsonify({'message': 'Demand created'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@api.route('/api/state', methods=['POST'])
def create_state():
    try:
        data = request.get_json()
        new_state = ElevatorState(floor=data['floor'], vacant=data['vacant'])
        db.session.add(new_state)
        db.session.commit()
        return jsonify({'message': 'State created'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@api.route('/api/demands', methods=['GET'])
def get_demand_table():
    try:
        tabledata = ElevatorDemand.query.all()
        demand_list = [{
            'id': demand.id,
            'floor': demand.floor,
            'timestamp': str(demand.timestamp)
        } for demand in tabledata]
        return jsonify({'demands': demand_list}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@api.route('/api/states', methods=['GET'])
def get_state():
    try:
        tabledata = ElevatorState.query.all()
        state_list = [{
            'id': state.id,
            'floor': state.floor,
            'vacant': state.vacant,
            'timestamp': str(state.timestamp)
        } for state in tabledata]
        return jsonify({'states': state_list}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
