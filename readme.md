# Notes 

Due to the short time frame of only 4 hours, I couldn't complete the one-hot encoding properly for the linear regression models. If more time had been given, I would have been able to finish it, and I'm confident I could have done it successfully. After that, I would have evaluated the performance of different machine learning models and selected the best one. Then, I would have fine-tuned the parameters to achieve the best results. Additionally, I would have needed to generate more data to train the models effectively. With a good prediction model in place, I would have developed a full-service API to deliver predictions in real time.

# Dev Test Description of my approach and constrains:

due the lack of real data, i'm gonna assume that each row in the elevator_demand table represents a call to the elevator, 
and each row in the elevator_state table represents the elevator's state at after the call.

In this scenario, I'll imagine a real-world situation where an elevator operates in a building with 1 to 13 floors. 
The elevator is called to a random floor and then travels to another random floor, taking approximately 3 seconds per floor. 
The simulation will span around 45 days. 
I'll also incorporate business rules: one of the floors (floor 6) is broken, so the elevator cannot stop there. 
Additionally, the building operates from 8:00 AM to 6:00 PM on weekdays only, meaning the elevator won't be called outside this time frame or during weekends.
Every morning the elevator is called to the first floor and at the end of the day it returns to the first floor.

# Instructions to run the project:
From the project folder:

### 1. Run Tests ensure the project is functioning correctly
```bash
pytest
```
### 2. host the database on localhost:5000. Set generate_random_test_data=True in the main.py file to populate the database with random use-case data.
```bash

python main.py
```
### 3. Extract important features, generate a DataFrame, and create visualizations based on the current data, these will be saved in the analytics folder.
```bash
python ./utils/process_data_features.py
```

### 4. Run the models with the extracted data to train and generate predictions.
```bash
python ./utils/train_model_frequency_based.py
python ./utils/train_model_time_sequential.py
```


# Descision making:
after analyzing the data, I see there are two approaches I can take to predict the next floor where the elevator will be called:

### Approach 1 (t,x = y) Frequency-Based Demand Prediction: 
Based on how often the same input (time_stamp, floor_state) is used for frequent travel, I can predict the output (floor_demand) where the elevator was originally called from. This way, I can position the elevator closer to that floor.

### Approach 2 (pt, px, py = t,x,y) Time-Sequential Demand Prediction: 
If I order the table historically, I can observe that there is always a previous travel before an elevator demand and state. I can store this in the table and create a relationship where the input consists of the previous travel (time_stamp, previous_demand, previous_state), and the output is the next travel (time_stamp, floor_demand, floor_state).

I would like to implement both approaches for the final decision-making process, aiming to minimize the travel time of the elevator prediction.









## Elevators
When an elevator is empty and not moving this is known as it's resting floor. 
The ideal resting floor to be positioned on depends on the likely next floor that the elevator will be called from.

We can build a prediction engine to predict the likely next floor based on historical demand, if we have the data.

The goal of this project is to model an elevator and save the data that could later be used to build a prediction engine for which floor is the best resting floor at any time
- When people call an elevator this is considered a demand
- When the elevator is vacant and not moving between floors, the current floor is considered its resting floor
- When the elevator is vacant, it can stay at the current position or move to a different floor
- The prediction model will determine what is the best floor to rest on


_The requirement isn't to complete this system but to start building a system that would feed into the training and prediction
of an ML system_

You will need to talk through your approach, how you modelled the data and why you thought that data was important, provide endpoints to collect the data and 
a means to store the data. Testing is important and will be used verify your system

## A note on AI generated code
This project isn't about writing code, AI can and will do that for you.
The next step in this process is to talk through your solution and the decisions you made to come to them. It makes for an awkward and rather boring interview reviewing chatgpt's solution.

If you use a tool to help you write code, that's fine, but we want to see _your_ thought process.

Provided under the chatgpt folder is the response you get back from chat4o. 
If your intention isn't to complete the project but to get an AI to spec it for you please, feel free to submit this instead of wasting OpenAI's server resources.


## Problem statement recap
This is a domain modeling problem to build a fit for purpose data storage with a focus on ai data ingestion
- Model the problem into a storage schema (SQL DB schema or whatever you prefer)
- CRUD some data
- Add some flair with a business rule or two
- Have the data in a suitable format to feed to a prediction training algorithm

---

#### To start
- Fork this repo and begin from there
- For your submission, PR into the main repo. We will review it, a offer any feedback and give you a pass / fail if it passes PR
- Don't spend more than 4 hours on this. Projects that pass PR are paid at the standard hourly rate

#### Marking
- You will be marked on how well your tests cover the code and how useful they would be in a prod system
- You will need to provide storage of some sort. This could be as simple as a sqlite or as complicated as a docker container with a migrations file
- Solutions will be marked against the position you are applying for, a Snr Dev will be expected to have a nearly complete solution and to have thought out the domain and built a schema to fit any issues that could arise 
A Jr. dev will be expected to provide a basic design and understand how ML systems like to ingest data


#### Trip-ups from the past
Below is a list of some things from previous submissions that haven't worked out
- Built a prediction engine
- Built a full website with bells and whistles
- Spent more than the time allowed (you won't get bonus points for creating an intricate solution, we want a fit for purpose solution)
- Overcomplicated the system mentally and failed to start
