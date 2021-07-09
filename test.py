import requests

data = {
	"calib": "http://127.0.0.1:5000/static/samples/calib.jpeg",
	"front": "http://127.0.0.1:5000/static/samples/front.jpeg",
	"side": "http://127.0.0.1:5000/static/samples/side.jpeg",
}
url = "http://127.0.0.1:5000/magic/api/"

r = requests.post(url, data=data)
print(r)
print(r.json())