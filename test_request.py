import requests

url = 'http://127.0.0.1:5000/predict'

# Email text to classify
email_text = "Subject: buy viagra now, special offer!"

# Send the POST request
response = requests.post(url, json={'text': email_text})

# Print the prediction result
print(response.json())  # Should print {"prediction": "Spam"}
