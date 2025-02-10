from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta, time
import random
from app.models import db, ElevatorDemand, ElevatorState

'''
In this scenario, I'll imagine a real-world situation where an elevator operates in a building with 1 to 13 floors. 
The elevator is called to a random floor and then travels to another random floor, taking approximately 3 seconds per floor. 
The simulation will span around 45 days. 
I'll also incorporate business rules: one of the floors (floor 6) is broken, so the elevator cannot stop there. 
Additionally, the building operates from 8:00 AM to 6:00 PM on weekdays only, meaning the elevator won't be called outside this time frame or during weekends.
Every morning the elevator is called to the first floor, and at the end of the day, it returns to the first floor.

Due to the lack of real data, I'm going to assume that each row in the elevator_demand table represents a call to the elevator, 
and each row in the elevator_state table represents the elevator's state after the call.
'''

DAYS_RANGE = 45
DATA_AMOUNT = (DAYS_RANGE * 50 ) + 500
TOTAL_FLOORS = 13
ELEVATOR_TRAVEL_TIME = 3  # seconds per floor
OPERATING_HOURS_START = time(8, 0)  # 8:00 AM
# OPERATING_HOURS_END = time(17, 0)  # 5:PM
OPERATING_CLOSING_TIME = time(18, 0)  # 6:PM
BROKEN_FLOOR = 6

def generate_random_elevator_data(num_rows=DATA_AMOUNT):
    print("Generating random elevator data...")

def insert_elevator_data_into_db():
    print("Inserting elevator data into the database...")

def delete_all_data_from_tables():
    print("Deleting all data from the elevator_demand and elevator_state tables...")