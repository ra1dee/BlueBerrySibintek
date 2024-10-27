from flask import Flask, request, jsonify, render_template, redirect, url_for
from selenium import webdriver

app = Flask(__name__)
data = ""

@app.route('/',methods = ["GET","POST"])
def index():
    global data
    if request.method == "POST":
        print(request.json)
        data = request.get_json()['text']
        if data is None:
            return "Invalid JSON", 400  # Return an error response
    return render_template("index.html",data=data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
