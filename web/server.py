from flask import Flask, render_template, request

# Use hacky relative module import until I create a proper package
import sys
sys.path.append("../gaitkeeper")
import load
import preprocess
import model

# Load reference features into memory
print("Loading reference dataset...")
# df_ref = preprocess.create_reference_data_features_from_fft_peaks(10)

# Create the application object
app = Flask(__name__)


@app.route('/', methods=["GET"])
def home_page():
    return render_template('index.html')


@app.route('/upload', methods=["POST"])
def upload_recording():
    df = load.read_form_data(request.form)
    # Placeholder for MVP...
    df_html = """<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Score</th>\n    </tr>\n    <tr>\n      <th>user_id</th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1</th>\n      <td>0.717070</td>\n    </tr>\n    <tr>\n      <th>40</th>\n      <td>0.570926</td>\n    </tr>\n    <tr>\n      <th>38</th>\n      <td>0.566829</td>\n    </tr>\n    <tr>\n      <th>26</th>\n      <td>0.562242</td>\n    </tr>\n    <tr>\n      <th>10</th>\n      <td>0.553433</td>\n    </tr>\n    <tr>\n      <th>35</th>\n      <td>0.551320</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>0.540472</td>\n    </tr>\n    <tr>\n      <th>21</th>\n      <td>0.537381</td>\n    </tr>\n    <tr>\n      <th>7</th>\n      <td>0.533832</td>\n    </tr>\n    <tr>\n      <th>29</th>\n      <td>0.530744</td>\n    </tr>\n    <tr>\n      <th>36</th>\n      <td>0.530096</td>\n    </tr>\n    <tr>\n      <th>46</th>\n      <td>0.521490</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>0.515982</td>\n    </tr>\n    <tr>\n      <th>27</th>\n      <td>0.514777</td>\n    </tr>\n    <tr>\n      <th>31</th>\n      <td>0.514268</td>\n    </tr>\n    <tr>\n      <th>28</th>\n      <td>0.513650</td>\n    </tr>\n    <tr>\n      <th>23</th>\n      <td>0.511082</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>0.509721</td>\n    </tr>\n    <tr>\n      <th>24</th>\n      <td>0.509693</td>\n    </tr>\n    <tr>\n      <th>33</th>\n      <td>0.499536</td>\n    </tr>\n    <tr>\n      <th>34</th>\n      <td>0.483492</td>\n    </tr>\n    <tr>\n      <th>43</th>\n      <td>0.483233</td>\n    </tr>\n    <tr>\n      <th>20</th>\n      <td>0.481084</td>\n    </tr>\n    <tr>\n      <th>16</th>\n      <td>0.472545</td>\n    </tr>\n    <tr>\n      <th>25</th>\n      <td>0.455413</td>\n    </tr>\n    <tr>\n      <th>19</th>\n      <td>0.451412</td>\n    </tr>\n    <tr>\n      <th>18</th>\n      <td>0.439560</td>\n    </tr>\n    <tr>\n      <th>37</th>\n      <td>0.411404</td>\n    </tr>\n    <tr>\n      <th>30</th>\n      <td>0.401884</td>\n    </tr>\n  </tbody>\n</table>"""

    # Return df.to_html and maybe a plot
    return render_template('result.html', model_output=df_html) #df.to_html())


if __name__ == "__main__":
    # Using a self-signed cert for SSL; need for sensor APIs to work
    app.run(host="0.0.0.0", debug=True, ssl_context=("cert.pem", "key.pem"))
