import pytest
from main import create_app
@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True

    with app.app_context():
        yield app.test_client()

def test_create_demand(client):
    print("\ntest_create_demand")
    response = client.post('/api/demand', json={'floor': 3})
    assert response.status_code == 201
    assert response.get_json() == {'message': 'Demand created'}

def test_get_demand(client):
    response = client.get('/api/demands')
    rows = response.get_json()["demands"]
    print("total rows", len(rows), "head", rows[:1])
    assert response.status_code == 200

def test_create_state(client):
    print("test_create_state")
    response = client.post('/api/state', json={'floor': 5, 'vacant': True})
    assert response.status_code == 201
    assert response.get_json() == {'message': 'State created'}

def test_get_state(client):
    response = client.get('/api/states')
    rows = response.get_json()["states"]
    print("total rows", len(rows), "head", rows[:1])
    assert response.status_code == 200

#for logging info
#pytest -s