# File: tests/test_flask_app.py

import unittest
from flask_app.app import app

class TestFlaskApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        """Test if home page is loading correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Car Price Predictor</title>', response.data)

    def test_prediction_route(self):
        """Test the predict route with sample car input."""
        sample_input = {
            'Company Name': 'Hyundai',
            'Car Name': 'i20',
            'Fuel Type': 'Petrol',
            'Make Year': '2019',
            'Owner Type': 'First',
            'Total_KM_Run': '35000',
            'Transmission Type': 'Manual',
            'Service Record': 'Yes',
            'Insurance': 'Comprehensive',
            'Registration Certificate': 'Original',
            'Accessories': 'Yes',
            'State Name': 'Delhi',
            'Tyre Condition': 'New'
        }

        response = self.client.post('/predict', data=sample_input)
        self.assertEqual(response.status_code, 200)

        # Check if response contains a price or success message
        self.assertTrue(
            b'Predicted Price' in response.data or b'₹' in response.data,
            "Response should contain 'Predicted Price' or currency symbol"
        )

if __name__ == '__main__':
    unittest.main()
