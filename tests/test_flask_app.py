import unittest
from flask_app.app import app

class TestFlaskApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Car Price Predictor</title>', response.data)

    def test_prediction_route(self):
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

        html = response.data.decode("utf-8")
        self.assertTrue(
            'Predicted Price' in html or '₹' in html,
            "Response should contain 'Predicted Price' or currency symbol"
        )

if __name__ == '__main__':
    unittest.main()
