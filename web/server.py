from flask import Flask, render_template, request

# Create the application object
app = Flask(__name__)


@app.route('/', methods=["GET"])
def home_page():
    return render_template('index.html')


@app.route('/upload', methods=["POST"])
def upload_recording():
    print(request.form["Gyroscope"])
    print(request.form["LinearAcceleration"])
    # Return df.to_html and maybe a plot
    return render_template('result.html', model_output="placeholder")


if __name__ == "__main__":
    # Using a self-signed cert for SSL; need for sensor APIs to work
    app.run(host="0.0.0.0", debug=True, ssl_context=("cert.pem", "key.pem"))
