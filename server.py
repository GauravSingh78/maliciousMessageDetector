import re
from urllib.parse import urlparse
from flask import Flask, render_template, request, jsonify
import pickle
from tld import get_tld
import os.path
from urllib.parse import urlparse
from googlesearch import search
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import requests

app = Flask(__name__)


# Load your trained model (replace 'your_trained_model.pkl' with your model file)
model = pickle.load(open('trained_model.pkl', 'rb'))


def preprocess_url(url):
    def having_ip_address(url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' 
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
        if match:
            return 1
        else:
            return 0

    def abnormal_url(url):
        hostname = urlparse(url).hostname
        hostname = str(hostname)
        match = re.search(hostname, url)
        if match:
            return 1
        else:
            return 0

    def google_index(url):
        site = search(url, 5)
        return 1 if site else 0

    def count_dot(url):
        count_dot = url.count('.')
        return count_dot

    def count_www(url):
        return url.count('www')

    def count_atrate(url):
        return url.count('@')

    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')

    def no_of_embed(url):
        urldir = urlparse(url).path
        return urldir.count('//')

    def shortening_service(url):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                          'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                          'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                          'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                          'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                          'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                          'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                          'tr\.im|link\.zip\.net',
                          url)
        if match:
            return 1
        else:
            return 0

    def count_https(url):
        return url.count('https')

    def count_http(url):
        return url.count('http')

    def count_per(url):
        return url.count('%')

    def count_ques(url):
        return url.count('?')

    def count_hyphen(url):
        return url.count('-')

    def count_equal(url):
        return url.count('=')

    def url_length(url):
        return len(str(url))

    def hostname_length(url):
        return len(urlparse(url).netloc)

    def suspicious_words(url):
        match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                          url)
        if match:
            return 1
        else:
            return 0

    def digit_count(url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
        return digits

    def letter_count(url):
        letters = 0
        for i in url:
            if i.isalpha():
                letters = letters + 1
        return letters

    def fd_length(url):
        urlpath = urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except:
            return 0

    def tld_length(tld):
        try:
            return len(tld)
        except:
            return -1

    use_ip = having_ip_address(url)
    abnormal = abnormal_url(url)
    google_index_val = google_index(url)
    dot_count = count_dot(url)
    www_count = count_www(url)
    atrate_count = count_atrate(url)
    dir_count = no_of_dir(url)
    embed_count = no_of_embed(url)
    short_url = shortening_service(url)
    https_count = count_https(url)
    http_count = count_http(url)
    per_count = count_per(url)
    ques_count = count_ques(url)
    hyphen_count = count_hyphen(url)
    equal_count = count_equal(url)
    length_url = url_length(url)
    length_hostname = hostname_length(url)
    sus_words = suspicious_words(url)
    digit_counts = digit_count(url)
    letter_counts = letter_count(url)
    first_dir_length = fd_length(url)
    tld = get_tld(url, fail_silently=True)
    tld_length_val = tld_length(tld)

    preprocessed_data = [
        use_ip, abnormal, dot_count, www_count, atrate_count, dir_count, embed_count,
        short_url, https_count, http_count, per_count, ques_count, hyphen_count, equal_count, length_url,
        length_hostname, sus_words, first_dir_length, tld_length_val, digit_counts, letter_counts
    ]

    return preprocessed_data


def majority_vote(predictions):
    vote_counter = Counter(predictions)
    majority_prediction = vote_counter.most_common(1)[0][0]
    return majority_prediction

@app.route('/', methods=['GET'])
def index():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = str(request.form['message'])

    # Loading trained models
    with open('logistic_regression_model.pkl', 'rb') as f:
        logistic_model = pickle.load(f)

    with open('naive_bayes_model.pkl', 'rb') as f:
        naive_bayes_model = pickle.load(f)

    with open('random_forest_model.pkl', 'rb') as f:
        random_forest_model = pickle.load(f)

    # Loading the CountVectorizer (ensure it's fitted with the same data used for training)
    with open('count_vectorizer.pkl', 'rb') as f:
        count_vectorizer = pickle.load(f)

    with open('accuracy_score.pkl', 'rb') as f:
        score = pickle.load(f)

    


    # Transforming input text into numerical features
    data_transformed = count_vectorizer.transform([data])

    # Making predictions
    logistic_prediction = logistic_model.predict(data_transformed)[0]
    naive_bayes_prediction = naive_bayes_model.predict(data_transformed)[0]
    random_forest_prediction = random_forest_model.predict(data_transformed)[0]

    log = logistic_model.predict_proba(data_transformed)[0][1]

    ran = random_forest_model.predict_proba(data_transformed)[0][1]

    nav = random_forest_model.predict_proba(data_transformed)[0][1]

    predicted_probability = (log + nav + ran)/3;
    # Finding majority prediction
    majority_prediction = majority_vote([
        logistic_prediction,
        naive_bayes_prediction,
        random_forest_prediction
    ])
    if majority_prediction == 1:
        d="S"
    else:
        d="H"
    predicted_probability=str(predicted_probability)
    return render_template('result.html', prediction_text=d+predicted_probability)
##############################################################


@app.route('/link', methods=['POST'])
def link():
    if request.method == 'POST':
        url = str(request.form['url'])
        with open('trained_model.pkl', 'rb') as f:
            model = pickle.load(f)
        response=""
        try:
            response=requests.get(url,verify=True)
            print(response)
        except:
            pass
        response=str(response)
        if response=="<Response [200]>":
            cert="G"
            pp=str(1)
            return render_template('result.html', prediction_text=cert+pp)
        else:
        # model = pickle.load(open('trained_model.pkl', 'rb'))

        # Preprocess the input URL
            preprocessed_url_data = preprocess_url(url)

        # Make predictions using the loaded model
            prediction = model.predict([preprocessed_url_data])[0]

            predicted_probabilities = model.predict_proba([preprocessed_url_data])
            pp = np.max(predicted_probabilities)


        # Convert the prediction code back to its label
            if prediction == 0:
                result = 'B'
            elif prediction == 1:
                result = 'D'
            elif prediction == 2:
                result = 'M'
            else:
                result = 'P'
            pp=str(pp)
            return render_template('result.html', prediction_text=result+pp)





if __name__ == '__main__':
    app.run(debug=True)
