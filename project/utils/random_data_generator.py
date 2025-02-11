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

# Check if the given timestamp falls within operating hours and weekdays
def is_within_operating_hours(timestamp):
    return (
        timestamp.weekday() < 5 and  # Weekdays only (Monday=0, Sunday=6)
        OPERATING_HOURS_START <= timestamp.time() <= OPERATING_CLOSING_TIME
    )

def get_random_floor_excluding(exclude_floor=-1):
    while True:
        floor = random.randint(1, TOTAL_FLOORS)
        if floor != BROKEN_FLOOR and floor != exclude_floor:
            break
    return floor

# Function to generate random timestamps and floor levels
def generate_random_elevator_data(num_rows=DATA_AMOUNT):
    print("Generating random elevator data...")
    demands = []
    states = []

    # generate morning and evening calls to the first floor 50 people at least
    for day in range(DAYS_RANGE * 50):
        
        random_seconds = random.randint(0, 0)
        current_day = datetime.now() - timedelta(days=day) + timedelta(seconds=random_seconds)
        if current_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            continue 

        # Start of the day (morning call to first floor)
        morning_time = current_day.replace(hour=OPERATING_HOURS_START.hour, minute=OPERATING_HOURS_START.minute, second=random_seconds, microsecond=0)
        demands.append(ElevatorDemand(timestamp=morning_time, floor=1))
        states.append(ElevatorState(timestamp=(morning_time + timedelta(seconds=ELEVATOR_TRAVEL_TIME)), 
                                   floor=get_random_floor_excluding(1), vacant=False))
        

        # End of the day (evening return to first floor)
        evening_time = current_day.replace(hour=OPERATING_CLOSING_TIME.hour, minute=OPERATING_CLOSING_TIME.minute, second=random_seconds, microsecond=0)
        demands.append(ElevatorDemand(timestamp=evening_time, floor=get_random_floor_excluding(1)))
        states.append(ElevatorState(timestamp=(evening_time + timedelta(seconds=ELEVATOR_TRAVEL_TIME)), 
                                   floor=1, vacant=True))
    
    # Generate a random timestamp within the last 45 days
    for _ in range(num_rows):
        while True:
            random_days = random.randint(0, DAYS_RANGE - 1)
            random_seconds = random.randint(0, 36000)  # Seconds in a day (10 hours from 8 AM to 6 PM)
            timestamp_demand = datetime.now() - timedelta(days=random_days, seconds=random_seconds)
            if is_within_operating_hours(timestamp_demand):
                break

        # Generate a random floor excluding the broken floor
        floor_demand = get_random_floor_excluding()
        # Generate a destination floor different from the demand floor and not broken
        floor_destination = get_random_floor_excluding(floor_demand)

        # Simulate the time it takes to reach the destination (3 seconds per floor)
        travel_time = abs(floor_destination - floor_demand) * ELEVATOR_TRAVEL_TIME
        timestamp_arrival = timestamp_demand + timedelta(seconds=travel_time)
        # Generate a random vacant status (for state)
        vacant_status = random.choice([True, False])

        demands.append(ElevatorDemand(timestamp=timestamp_demand, floor=floor_demand))
        states.append(ElevatorState(timestamp=timestamp_arrival, floor=floor_destination, vacant=vacant_status))
    
    return demands, states

# insert the generated data into the table
def insert_elevator_data_into_db():
    print("Inserting elevator data into the database...")
    try:
        # Generate random data
        demands, states = generate_random_elevator_data(DATA_AMOUNT)
        # Bulk insert elevator demand data into the database
        db.session.bulk_save_objects(demands)
        db.session.commit()
        # Bulk insert elevator state data into the database
        db.session.bulk_save_objects(states)
        db.session.commit()

        print(f"Successfully inserted {len(demands)} rows.")
    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"Error inserting data: {e}")

def delete_all_data_from_tables():
    print("Deleting all data from the elevator_demand and elevator_state tables...")
    try:
        db.session.query(ElevatorDemand).delete()
        db.session.query(ElevatorState).delete()
        db.session.commit()
        print("All data has been successfully deleted from the elevator_demand and elevator_state tables.")
    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"Error deleting data: {e}")