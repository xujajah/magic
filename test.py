import requests

data = {
	"calib": "https://127.0.0.1:5000/static/samples/calib.jpeg",
	"front": "https://127.0.0.1:5000/static/samples/front.jpeg",
	"side": "https://127.0.0.1:5000/static/samples/side.jpeg",
}
url = "https://127.0.0.1:5000/api/magic"

r = requests.post(url, data=data, verify=False)
print(r)
print(r.json())
