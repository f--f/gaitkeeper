# [Gᴀɪtkeeper](https://gaitkeeper.live): *authenticate with your walk*

This project is a gait-based biometric authentication proof-of-concept with the goal authenticating call center customers using characteristics of their gait or walking pattern. It allows users to record their walks using measurements from ubiquitous smartphone motion sensors (accelerometer and gyroscope) and allows agents to compare these recordings against a reference to verify a customer's identity using a machine learning model. You can access the demo at https://gaitkeeper.live.

Gᴀɪtkeeper was developed in a three-week period as part of [Insight Data Science](https://www.insightdatascience.com/). The original IDNet reference dataset used to bootstrap the initial model is available [here](http://signet.dei.unipd.it/research/human-sensing/).

## Approach

1. Phone sensor APIs are used to record instantaneous accelerometer and gyrometer data at a frequency of 60 Hz, which is ingested into a PostgreSQL database. 
2. Since the sensors sample at varying rates, the sensor data for a walk is cleaned by resampling and interpolating missing values. The data is also normalized so it is invariant to initial device orientation.
3. The resulting walk data is broken up into fixed-length samples and fed through a convolutional neural network incorporating 1D filters which acts as a feature extraction pipeline to transform each sample into a lower-dimensional feature embedding.
4. Feature embeddings are compared using cosine similarity to arrive at a final similarity score.

## Screenshots

<p float="left" align="middle">
  <img src="./screenshots/record.png" width="322" hspace="20"/>
  <img src="./screenshots/compare.gif" width="322" hspace="20"/>
</p>

## Developed With

- *Data Cleaning/Modelling*: pandas, scipy, scikit-learn, PyTorch
- *Data Visualization*: matplotlib, seaborn
- *Backend*: Flask, gunicorn & NGINX, PostgreSQL (psycopg2), AWS (RDS & EC2)
- *Frontend*: Plotly.js, [progressbar.js](https://kimmobrunfeldt.github.io/progressbar.js/), [NoSleep.js](https://github.com/richtr/NoSleep.js), Bulma (CSS)
