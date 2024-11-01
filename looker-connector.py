# requirements.txt
flask==2.0.1
pandas==1.4.2
numpy==1.21.5
scipy==1.7.3
scikit-learn==1.0.2
plotly==5.6.0
python-dotenv==0.19.2
pyjwt==2.3.0
redis==4.1.4
gunicorn==20.1.0
requests==2.27.1

# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import jwt
import redis
import os
from dotenv import load_dotenv
from functools import wraps

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')

# Initialize Redis for rate limiting
redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))

def rate_limit(limit=100, per=3600):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                return jsonify({'error': 'API key required'}), 401
            
            redis_key = f'rate_limit:{api_key}'
            current = redis_client.get(redis_key)
            
            if current is None:
                redis_client.setex(redis_key, per, 1)
            elif int(current) >= limit:
                return jsonify({'error': 'Rate limit exceeded'}), 429
            else:
                redis_client.incr(redis_key)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

class DataPreprocessor:
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
    def handle_missing_values(self, data):
        """Handle missing values using various strategies"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # For numeric columns, fill with median
        for col in numeric_cols:
            data[col].fillna(data[col].median(), inplace=True)
            
        # For categorical columns, fill with mode
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)
            
        return data
    
    def remove_outliers(self, data, method='zscore', threshold=3):
        """Remove outliers using various methods"""
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
            data = data[(z_scores < threshold).all(axis=1)]
        elif method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            yhat = iso_forest.fit_predict(data.select_dtypes(include=[np.number]))
            data = data[yhat == 1]
        return data
    
    def encode_categorical(self, data):
        """Encode categorical variables"""
        return pd.get_dummies(data)

class DataAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        
    def detect_anomalies(self, data, column):
        """Detect anomalies using multiple methods"""
        z_scores = stats.zscore(data[column])
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        yhat = iso_forest.fit_predict(data[column].values.reshape(-1, 1))
        
        return {
            'zscore_anomalies': np.abs(z_scores) > 3,
            'isolation_forest_anomalies': yhat == -1
        }
    
    def find_correlations(self, data, threshold=0.7):
        """Find correlations with additional metadata"""
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    strong_corrs.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j],
                        'strength': 'strong positive' if corr_matrix.iloc[i, j] > 0 else 'strong negative'
                    })
        return strong_corrs
    
    def identify_trends(self, data, time_col, value_col):
        """Enhanced trend analysis"""
        if time_col not in data.columns or value_col not in data.columns:
            return None
            
        data = data.sort_values(time_col)
        x = np.arange(len(data))
        y = data[value_col].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Moving averages
        ma_7 = data[value_col].rolling(window=7).mean()
        ma_30 = data[value_col].rolling(window=30).mean()
        
        # Seasonality detection
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomposition = seasonal_decompose(data[value_col], period=7)
            has_seasonality = np.std(decomposition.seasonal) > 0.1 * np.std(data[value_col])
        except:
            has_seasonality = None
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'is_significant': p_value < 0.05,
            'moving_avg_7': list(ma_7.fillna(0)),
            'moving_avg_30': list(ma_30.fillna(0)),
            'has_seasonality': has_seasonality
        }
    
    def segment_data(self, data, n_clusters=3):
        """Enhanced data segmentation"""
        numeric_data = data.select_dtypes(include=[np.number])
        scaled_data = self.scaler.fit_transform(numeric_data)
        reduced_data = self.pca.fit_transform(scaled_data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced_data)
        
        # Calculate cluster characteristics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_data = numeric_data[clusters == i]
            stats = {
                'size': len(cluster_data),
                'center': cluster_data.mean().to_dict(),
                'variance': cluster_data.var().to_dict()
            }
            cluster_stats.append(stats)
        
        return {
            'cluster_assignments': list(clusters),
            'cluster_stats': cluster_stats,
            'pca_components': self.pca.components_.tolist(),
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist()
        }

class VisualizationGenerator:
    @staticmethod
    def create_correlation_heatmap(data):
        """Generate correlation heatmap"""
        corr_matrix = data.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       title="Correlation Heatmap")
        return fig.to_json()
    
    @staticmethod
    def create_trend_visualization(data, time_col, value_col):
        """Generate trend visualization"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[time_col], y=data[value_col],
                               mode='lines', name='Original Data'))
        fig.add_trace(go.Scatter(x=data[time_col], 
                               y=data[value_col].rolling(window=7).mean(),
                               mode='lines', name='7-day MA'))
        return fig.to_json()
    
    @staticmethod
    def create_cluster_visualization(data, clusters):
        """Generate cluster visualization"""
        pca = PCA(n_components=2)
        coords = pca.fit_transform(data.select_dtypes(include=[np.number]))
        
        fig = px.scatter(x=coords[:, 0], y=coords[:, 1], 
                        color=clusters,
                        labels={'color': 'Cluster'},
                        title='Cluster Visualization (PCA)')
        return fig.to_json()

@app.route('/analyze', methods=['POST'])
@rate_limit(limit=100, per=3600)
def analyze_data():
    try:
        # Get data from request
        data = pd.DataFrame(request.json['data'])
        
        # Initialize components
        preprocessor = DataPreprocessor()
        analyzer = DataAnalyzer()
        visualizer = VisualizationGenerator()
        
        # Preprocess data
        data = preprocessor.handle_missing_values(data)
        data = preprocessor.remove_outliers(data)
        
        # Initialize results dictionary
        insights = {
            'summary_stats': {},
            'anomalies': {},
            'correlations': [],
            'trends': {},
            'segments': {},
            'visualizations': {}
        }
        
        # Generate insights
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # Summary statistics
        insights['summary_stats'] = data.describe().to_dict()
        
        # Anomaly detection
        for column in numeric_columns:
            insights['anomalies'][column] = analyzer.detect_anomalies(data, column)
        
        # Correlation analysis
        insights['correlations'] = analyzer.find_correlations(data)
        insights['visualizations']['correlation_heatmap'] = visualizer.create_correlation_heatmap(data)
        
        # Time series analysis
        date_columns = data.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            for value_col in numeric_columns:
                trend = analyzer.identify_trends(data, date_columns[0], value_col)
                if trend:
                    insights['trends'][value_col] = trend
                    insights['visualizations'][f'trend_{value_col}'] = visualizer.create_trend_visualization(
                        data, date_columns[0], value_col)
        
        # Segmentation
        if len(numeric_columns) >= 2:
            segments = analyzer.segment_data(data)
            insights['segments'] = segments
            insights['visualizations']['clusters'] = visualizer.create_cluster_visualization(
                data, segments['cluster_assignments'])
        
        return jsonify({
            'status': 'success',
            'insights': insights
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=8080)

# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./
