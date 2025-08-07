import unittest
from flask_app.app import app

class TestFlaskApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Car Price Prediction</title>', response.data)

    def test_prediction_route(self):
        # Using values that match your HTML form (e.g., "1" and "0" for booleans)
        form_data = {
            'Company': 'TATA',
            'CarName': 'Nexon',
            'Fuel Type': 'Petrol',
            'Tyre Condition': 'New',
            'Make Year': '2020',
            'Owner Type': 'First',
            'Total_KM_Run': '10000',
            'Transmission Type': 'Manual',
            'Service Record': '1',
            'Insurance': '1',
            'Registration Certificate': '1',
            'Accessories': 'Music System',
            'StateName': 'Delhi'
        }

        response = self.client.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction Result', response.data)

if __name__ == '__main__':
    unittest.main()
