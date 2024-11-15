import gradio as gr
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import pandas as pd
import os

def load_data(file):
    # Ensure the file is not None
    if file is None:
        raise ValueError("No file uploaded. Please upload a CSV or XLSX file.")

    # Determine file type and load accordingly
    file_extension = os.path.splitext(file.name)[-1].lower()
    if file_extension == '.csv':
        df = pd.read_csv(file.name)
    elif file_extension == '.xlsx':
        df = pd.read_excel(file.name)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")
    
    return df

def analyze_customer_segments(file):
    # Load the dataset
    df = load_data(file)
    
    # Check for date column and parse it
    date_col = 'Date' if 'Date' in df.columns else 'InvoiceDate' if 'InvoiceDate' in df.columns else None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        raise ValueError("Date column not found. Please ensure 'Date' or 'InvoiceDate' is present.")

    # Calculate reference date for Recency
    reference_date = df[date_col].max() + timedelta(days=1)

    # Determine column names for grouping
    customer_col = 'Customer ID' if 'Customer ID' in df.columns else 'CustomerID'
    transaction_col = 'Transaction ID' if 'Transaction ID' in df.columns else 'InvoiceNo'
    total_amount_col = 'Total Amount' if 'Total Amount' in df.columns else 'TotalPrice'

    # Calculate 'TotalPrice' if necessary
    if total_amount_col not in df.columns:
        if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
            total_amount_col = 'TotalPrice'
        else:
            raise ValueError("Columns for calculating 'TotalPrice' not found. Ensure 'Quantity' and 'UnitPrice' are present.")

    # Calculate RFM metrics
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,
        transaction_col: 'nunique',
        total_amount_col: 'sum'
    }).rename(columns={date_col: 'Recency', transaction_col: 'Frequency', total_amount_col: 'Monetary'}).reset_index()

    # RFM scoring and segmentation
    rfm['recency_score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm['frequency_score'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['monetary_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['RFM_Segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

    # Map RFM segments to meaningful labels
    seg_map = {
        r'[1-2][1-2]': 'Hibernating',
        r'[1-2][3-4]': 'At Risk',
        r'[1-2]5': "Can't Lose",
        r'3[1-2]': 'About to Sleep',
        r'33': 'Need Attention',
        r'[3-4][4-5]': 'Loyal Customers',
        r'41': 'Promising',
        r'51': 'New Customers',
        r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]': 'Champions'
    }
    rfm['RFM_Segment_Label'] = rfm['RFM_Segment'].replace(seg_map, regex=True)

    # Summary of RFM segmentation
    rfm_summary = rfm['RFM_Segment_Label'].value_counts().reset_index()
    rfm_summary.columns = ['Segment', 'Customer Count']

    # Create and save the treemap of RFM segments
    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=rfm_summary['Customer Count'], label=rfm_summary['Segment'], alpha=0.8, color=sns.color_palette("Set3"))
    plt.title("RFM Segment Treemap")
    plt.axis('off')
    plt.savefig("rfm_treemap.png")
    
    # Filter high-value segments and create a copy to avoid warnings
    high_value_customers = rfm[rfm['RFM_Segment_Label'].isin(['Champions', 'Loyal Customers'])].copy()

    # Scale the RFM features for clustering
    scaler = StandardScaler()
    high_value_scaled = scaler.fit_transform(high_value_customers[['Recency', 'Frequency', 'Monetary']])

    # Apply Gaussian Mixture Model for clustering
    gmm = GaussianMixture(n_components=4, covariance_type='tied', random_state=42)
    high_value_customers['GMM_Sub_Cluster'] = gmm.fit_predict(high_value_scaled)

    # Plotting the sub-clusters within high-value segments
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=high_value_customers, x='Recency', y='Monetary', hue='GMM_Sub_Cluster', palette='viridis', s=100, alpha=0.7)
    plt.title("Sub-Clusters within High-Value Segments")
    plt.xlabel("Recency")
    plt.ylabel("Monetary")
    plt.legend(title="GMM Sub-Cluster")
    plt.savefig("sub_clusters_plot.png")  # Save plot as an image

    # Recommendations based on RFM Segments
    recommendations = []
    for segment in rfm_summary['Segment']:
        if segment == 'Champions':
            recommendations.append(f"{segment}: Reward these loyal customers with exclusive offers or early access to new products.")
        elif segment == 'Loyal Customers':
            recommendations.append(f"{segment}: Build deeper engagement through loyalty programs or personalized discounts.")
        elif segment == 'At Risk':
            recommendations.append(f"{segment}: Win back these customers with targeted re-engagement campaigns or special discounts.")
        elif segment == "Can't Lose":
            recommendations.append(f"{segment}: These are high-value customers who haven't purchased recently. Re-engage them with special offers.")
        elif segment == 'Hibernating':
            recommendations.append(f"{segment}: Reactivate these customers through reminder emails or re-targeting ads.")
        elif segment == 'New Customers':
            recommendations.append(f"{segment}: Encourage these customers to make a repeat purchase with welcome offers.")
        elif segment == 'Potential Loyalists':
            recommendations.append(f"{segment}: Encourage loyalty by providing excellent service and personalized recommendations.")
        elif segment == 'Need Attention':
            recommendations.append(f"{segment}: Provide special attention or incentives to increase engagement.")
        elif segment == 'Promising':
            recommendations.append(f"{segment}: Nurture these customers with personalized offers to help them become loyal customers.")
        elif segment == 'About to Sleep':
            recommendations.append(f"{segment}: Re-engage these customers with targeted marketing campaigns before they churn.")

    # Generate recommendations based on GMM sub-clusters
    sub_cluster_summary = high_value_customers.groupby('GMM_Sub_Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()

    sub_cluster_recommendations = []
    for index, row in sub_cluster_summary.iterrows():
        if row['Recency'] < 30 and row['Frequency'] > 10 and row['Monetary'] > 5000:
            sub_cluster_recommendations.append(f"Sub-Cluster {row['GMM_Sub_Cluster']}: Reward these high-value recent purchasers with loyalty rewards and exclusive product previews.")
        elif row['Recency'] > 90 and row['Frequency'] > 5:
            sub_cluster_recommendations.append(f"Sub-Cluster {row['GMM_Sub_Cluster']}: Re-engage these customers who purchase frequently but have not purchased recently with special offers.")
        elif row['Monetary'] < 1000:
            sub_cluster_recommendations.append(f"Sub-Cluster {row['GMM_Sub_Cluster']}: Encourage these lower-value customers to increase spending through bundle deals or discounts.")
        else:
            sub_cluster_recommendations.append(f"Sub-Cluster {row['GMM_Sub_Cluster']}: Maintain engagement through personalized communication and targeted marketing.")

    recommendations_text = "\n".join(recommendations + sub_cluster_recommendations)
    
    return rfm_summary, sub_cluster_summary, "rfm_treemap.png", "sub_clusters_plot.png", recommendations_text

# Define the Gradio interface
interface = gr.Interface(
    fn=analyze_customer_segments,
    inputs="file",
    outputs=["dataframe", "dataframe", "image", "image", "text"],
    title="Customer Segmentation Analysis with Recommendations",
    description="Upload a CSV or XLSX file to perform RFM segmentation and clustering analysis, and get recommendations."
)

interface.launch()
