from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load your trained models and preprocessors
try:
    with open('models/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open('models/pca_cluster.pkl', 'rb') as f:
        pca_cluster = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/content_encoder.pkl', 'rb') as f:
        content_encoder = pickle.load(f)
    with open('models/product_features.pkl', 'rb') as f:
        product_features_encoded = pickle.load(f)
    with open('models/product_data.pkl', 'rb') as f:
        product_data = pickle.load(f)
    with open('models/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    print("All models loaded successfully!")
except FileNotFoundError as e:
    print(f"Model files not found: {e}")
    print("Please run create_models.py first!")
    xgb_model = None
    kmeans_model = None
    pca_cluster = None
    scaler = None
    content_encoder = None
    product_features_encoded = None
    product_data = None
    feature_columns = None
#global variables
categorical_mappings = {
            'Gender': ['Male',"Female"], 
            'Income': ['Low', 'Medium',"High"], 
            'Customer_Segment': ['Premium', 'Regular',"New"],
            'Product_Category': ['Clothing', 'Electronics', 'Grocery', 'Home Decor','Books'],
            'Product_Brand': ['Pepsi', 'Coca-Cola', 'Samsung', 'Zara',
                'HarperCollins', 'Sony', 'Bed Bath & Beyond', 'Adidas', 'Home Depot',
                'Nike', 'Penguin Books', 'Random House', 'Nestle', 'Apple',
                'IKEA', 'Whirepool', 'Mitsubhisi', 'BlueStar'],
            'Product_Type': ['Water', 'Smartphone', 'Non-Fiction', 'Fiction', 'Juice', 'Television',
                'T-shirt', 'Decorations', 'Shoes', 'Tablet',
                'Soft Drink', 'Furniture', 'Fridge', 'Mitsubishi 1.5 Ton 3 Star Split AC',
                'Thriller', 'Kitchen', 'Coffee', "Children's", 'Jeans', 'Shirt',
                'Dress', 'Shorts', 'Headphones', 'Lighting', 'Chocolate', 'Literature',
                'Bathroom', 'Bedding', 'Jacket', 'Laptop', 'Tools', 'Snacks', 'BlueStar AC'],
            'Feedback': ['Bad', 'Excellent', 'Good','Average'], 
            'Shipping_Method': ['Express', 'Standard',"Same-Day"],
            'Payment_Method': ['Credit Card', 'Debit Card', 'PayPal','Cash'], 
            'Order_Status': ['Processed', 'Shipped',"Delivered","Pending"] 
        }
numeric_cols = ['Age', 'Total_Purchases', 'Amount', 'Total_Amount', 'Ratings']
categorical_cols = ['Gender', 'Income', 'Month', 'Customer_Segment', 'Product_Category',
'Product_Brand', 'Product_Type', 'Feedback', 'Shipping_Method','Payment_Method', 'Order_Status']

def common_preprocessing(user_data):
    # Create input dataframe
    user_df = pd.DataFrame([user_data])       
    # Map month to number
    month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
    user_df['Month_Num'] = user_df['Month'].map(month_map)
            

    # Scale numerical features using the same scaler used in training
    user_num_scaled = scaler.transform(user_df[numeric_cols])
            
    #One-hot encode categorical features
    user_encoded = pd.get_dummies(user_df, columns=categorical_cols, drop_first=True)
    return user_num_scaled,user_encoded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.json
        print(f"Received data: {data}")
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data received'
            })
        
        predictions = {}
        errors = []
        user_num_scaled,user_encoded=common_preprocessing(data)
        # 1. Content-Based Filtering using Cosine Similarity
        try:
            if content_encoder is not None and product_features_encoded is not None:
                content_recommendations = get_content_based_recommendations(data)
                predictions['content_recommendations'] = content_recommendations
            else:
                errors.append('Content-based filtering model not available')
        except Exception as e:
            errors.append(f'Content-based filtering error: {str(e)}')
        
        # 2. Price Prediction using XGBoost
        try:
            if xgb_model is not None and scaler is not None:
                price_prediction = get_price_prediction(user_encoded)
                predictions['predicted_spending'] = price_prediction
            else:
                errors.append('XGBoost model not available')
        except Exception as e:
            errors.append(f'Price prediction error: {str(e)}')
        
        # 3. Cluster Prediction using K-means
        try:
            if kmeans_model is not None and pca_cluster is not None:
                cluster_prediction = get_cluster_prediction(user_encoded, user_num_scaled)
                predictions['cluster'] = cluster_prediction['cluster']
                predictions['cluster_recommendations'] = cluster_prediction['recommendations']
            else:
                errors.append('K-means model not available')
        except Exception as e:
            errors.append(f'Cluster prediction error: {str(e)}')
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'warnings': errors if errors else None
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_content_based_recommendations(user_data):
    """Content-based filtering using cosine similarity"""
    try:
        #Prepare user input for content-based filtering
        user_features = [[
            user_data['Gender'],
            user_data['Income'],
            user_data['Month'],
            user_data['Customer_Segment'],
            user_data['Product_Category'],
            user_data['Product_Brand'],
            user_data['Product_Type'],
            user_data['Feedback'],
            user_data['Shipping_Method'],
            user_data['Payment_Method'],
            user_data['Order_Status']
        ]]
        
        # Transform user input
        user_encoded = content_encoder.transform(user_features)
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(user_encoded, product_features_encoded).flatten()
        
        # Get top 5 most similar products
        top_indices = similarity_scores.argsort()[-6:-1][::-1]  # Get top 5 (excluding the same input)
        
        recommendations = []
        for idx in top_indices:
            if idx < len(product_data):
                product = product_data.iloc[idx]
                recommendations.append({
                    'product': product.get('products', 'N/A'),
                    'category': product.get('Product_Category', 'N/A'),
                    'brand': product.get('Product_Brand', 'N/A'),
                    'type': product.get('Product_Type', 'N/A'),
                    'similarity': float(similarity_scores[idx])
                })
        
        return recommendations[:5]  # Return top 5
        
    except Exception as e:
        print(f"Error in content-based filtering: {str(e)}")
        return []

def get_price_prediction(user_data):
    """Price prediction using XGBoost regression"""
    try:
        input_data=user_data.copy()
        input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])        
        # Ensure all required columns are present and in correct order
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Select only the columns used during training in the correct order
        input_data = input_data[feature_columns]
        
        # Make prediction
        log_pred = xgb_model.predict(input_data)
        spend_pred = 100 * np.exp(1 + log_pred[0])
        
        return f"${spend_pred:.2f}"
        
    except Exception as e:
        print(f"Error in price prediction: {str(e)}")
        return "Error in prediction"

def get_cluster_prediction(user_encoded, user_num_scaled):
    """Cluster prediction using K-means with proper preprocessing"""
    try:
        #columns that should be present (excluding the original categorical columns and other non-feature columns)
        feature_cols = [col for col in user_encoded.columns if col not in categorical_cols and col not in ['Month', 'Year']]        
        #template with all possible categorical encodings
        expected_categorical_features = [] 
        #Create a proper feature vector
        combined_features = np.array(user_num_scaled[0])  # Start with numeric features                
        #Add categorical features in the expected order
        for cat_col, possible_values in categorical_mappings.items():
            for value in possible_values:
                col_name = f"{cat_col}_{value}"
                if col_name in user_encoded.columns:
                    combined_features = np.append(combined_features, user_encoded[col_name].values[0])
                else:
                    combined_features = np.append(combined_features, 0)                
        #Reshape for prediction
        combined_features = combined_features.reshape(1, -1)
        print(f"Combined features shape: {combined_features.shape}")
        print(f"Expected PCA input features: {pca_cluster.n_features_in_ if hasattr(pca_cluster, 'n_features_in_') else 'Unknown'}")                
        
        expected_features = pca_cluster.n_features_in_ if hasattr(pca_cluster, 'n_features_in_') else 77
                
        if combined_features.shape[1] < expected_features:
            padding = np.zeros((1, expected_features - combined_features.shape[1]))
            combined_features = np.hstack((combined_features, padding))
        elif combined_features.shape[1] > expected_features:
            combined_features = combined_features[:, :expected_features]
        
        # Apply PCA transformation
        user_reduced = pca_cluster.transform(combined_features)
        
        # Predict cluster using the PCA-transformed data
        predicted_cluster = kmeans_model.predict(user_reduced)
        
        print(f"This user belongs to Cluster: {predicted_cluster[0]}")
        
        # Get cluster-based recommendations
        cluster_recommendations = get_cluster_recommendations(predicted_cluster[0])
        
        return {
            'cluster': int(predicted_cluster[0]),
            'recommendations': cluster_recommendations
        }
        
    except Exception as e:
        print(f"Error in cluster prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'cluster': -1, 'recommendations': {}}

def get_cluster_recommendations(cluster_id):
    """Get product recommendations based on cluster"""
    cluster_recommendations = {
        0: {
            'description': 'Regular and House appliance Customers',
            'brands': ['Bed Bath & Beyond', 'IKEA', 'Zara', 'Home Depot', 'Pepsi'],
            'categories': ['Home Decor', 'Electronics', 'Clothing', 'Grocery'],
            'products': ['Smartphone', 'Water', 'Furniture', 'Fridge', 'Decorations'],
            'characteristics': 'Prefers value for money, shops regularly for essentials and home appliances'
        },
        1: {
            'description': 'Leisure and Lifestyle Shoppers',
            'brands': ['Pepsi', 'IKEA', 'Home Depot', 'Zara', 'Bed Bath & Beyond'],
            'categories': ['Grocery', 'Home Decor', 'Clothing', 'Electronics'],
            'products': ['Water', 'Furniture', 'Soft Drink', 'Fiction', 'Decorations'],
            'characteristics': 'Interested in entertainment and snacks'
        },
        2: {
            'description': 'Quality-Conscious Frequent Buyers',
            'brands': ['Pepsi', 'Nestle', 'Zara', 'Penguin Books', 'Nike'],
            'categories': ['Grocery', 'Electronics', 'Clothing', 'Home Decoration'],
            'products': ['Water', 'Tablet', 'Non-Fiction', 'Shorts', 'Furniture'],
            'characteristics': 'High spending power, brand conscious, quality over price'
        },
        3: {
            'description': 'Semi-Premium and Budget Shoppers',
            'brands': ['Random House', 'Sony', 'HarperCollins', 'Samsung', 'Pepsi'],
            'categories': ['Electronics', 'Clothing', 'Grocery', 'Home Decoration'],
            'products': ['Smartphone', 'Non-Fiction', 'Fiction', 'T-shirt', 'Juice'],
            'characteristics': 'Cost-conscious but willing to spend on quality products'
        },
        4: {
            'description': 'Premium Segment',
            'brands': ['Apple', 'Pepsi', 'Samsung', 'Coca-Cola', 'Sony'],
            'categories': ['Electronics', 'Grocery', 'Clothing', 'Home Decoration'],
            'products': ['Soft Drink', 'Non-Fiction', 'Fiction', 'Smartphone', 'Tablet'],
            'characteristics': 'Premium segment, luxury-focused, exclusive products'
        }
    }
    
    return cluster_recommendations.get(cluster_id, {
        'description': 'General Customer Segment',
        'brands': ['Various Brands'],
        'categories': ['General Categories'],
        'products': ['General Products'],
        'characteristics': 'Mixed shopping behavior'
    })

if __name__ == '__main__':
    app.run(debug=True)