import pandas as pd
import numpy as np
import re
import socket
import ssl
import joblib
import whois
import datetime
import urllib.parse
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import requests
from bs4 import BeautifulSoup

def has_ip(url):
    # Detect if URL contains an IP address instead of domain name
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc
        ipaddress.ip_address(host)
        return 1
    except:
        return 0

def count_dots(url):
    return url.count('.')

def has_at_symbol(url):
    return 1 if '@' in url else 0

def url_length(url):
    return len(url)

def is_shortened(url):
    # Common URL shorteners
    shorteners = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'is.gd', 'buff.ly',
                  'adf.ly', 'bit.do', 'cutt.ly', 'tiny.cc', 'fb.me']
    for shortener in shorteners:
        if shortener in url:
            return 1
    return 0

def domain_age(domain):
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):  # Sometimes a list of dates
            creation_date = creation_date[0]
        if creation_date is None:
            return -1
        age = (datetime.datetime.now() - creation_date).days
        return age
    except:
        return -1

def ssl_certificate_status(domain):
    # Check if site supports https with valid SSL certificate
    try:
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                # Could add more detailed validation here
                return 1
    except:
        return 0

def domain_expiration_period(domain):
    # Get days till domain expiration
    try:
        w = whois.whois(domain)
        expiration_date = w.expiration_date
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        if expiration_date is None:
            return -1
        days_left = (expiration_date - datetime.datetime.now()).days
        return days_left
    except:
        return -1

def count_tags(html_content, tag_name):
    soup = BeautifulSoup(html_content, 'html.parser')
    return len(soup.find_all(tag_name))

def has_external_form_action(html_content, domain):
    soup = BeautifulSoup(html_content, 'html.parser')
    forms = soup.find_all('form')
    for form in forms:
        action = form.get('action')
        if action:
            parsed_action = urllib.parse.urlparse(action)
            # If action has a netloc and is not the same domain, external action
            if parsed_action.netloc and domain not in parsed_action.netloc:
                return 1
    return 0

def extract_html_js_features(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return (0, 0, 0)
        html = response.text
        count_form = count_tags(html, 'form')
        external_form_action = has_external_form_action(html, urllib.parse.urlparse(url).netloc)
        count_iframe = count_tags(html, 'iframe')
        return count_form, external_form_action, count_iframe
    except:
        return (0, 0, 0)

def extract_features(url):
    features = {}
    features['url_length'] = url_length(url)
    features['dot_count'] = count_dots(url)
    features['has_at'] = has_at_symbol(url)
    features['has_ip'] = has_ip(url)
    features['is_shortened'] = is_shortened(url)
    
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    features['domain_age'] = domain_age(domain)
    features['ssl_status'] = ssl_certificate_status(domain)
    features['domain_expiration'] = domain_expiration_period(domain)
    
    f_count, ext_form, iframe_count = extract_html_js_features(url)
    features['form_count'] = f_count
    features['external_form_action'] = ext_form
    features['iframe_count'] = iframe_count
    
    # Handle missing numeric values - domain_age and domain_expiration might be -1 if unknown
    for key in ['domain_age', 'domain_expiration']:
        if features[key] == -1:
            features[key] = 0

    return features

def prepare_dataset(df):
    # Dataframe must have columns: 'url' and 'label'
    feature_list = []
    for url in df['url']:
        try:
            features = extract_features(url)
        except Exception as e:
            # On failure, use default zeros
            features = {'url_length':0, 'dot_count':0, 'has_at':0, 'has_ip':0, 'is_shortened':0,
                        'domain_age':0, 'ssl_status':0, 'domain_expiration':0,
                        'form_count':0, 'external_form_action':0, 'iframe_count':0}
        feature_list.append(features)
    features_df = pd.DataFrame(feature_list)
    return features_df

def main():
    print("Loading dataset...")
    df = pd.read_csv('../data/phishing_dataset.csv')
    # Assuming dataset has columns 'url' and 'label' where label is 1 (phishing) or 0 (legitimate)
    
    print("Extracting features. This may take some time due to network calls...")
    X = prepare_dataset(df)
    y = df['label']

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    print("Predicting test data...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("Saving model to 'xgb_phishing_model.joblib'...")
    joblib.dump(model, 'xgb_phishing_model.joblib')

if __name__ == "__main__":
    main()

