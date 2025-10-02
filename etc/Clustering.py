import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def get_db_conn():
    """Establishes and returns a connection to the PostgreSQL database."""
    
    engine = create_engine("postgresql://ssian:mysecretpassword@localhost:5432/piscineds")

    return engine


'''Keep only the "purchase" data of "event_type" column'''
def load_df_sql(schema, table, engine):
    """Load customers group by event_type into a df, return the df"""

    load_sql = f"""SELECT c.event_time, c.user_id, sum(c.price) as basket_price FROM {schema}.{table} c
                        WHERE c.event_type = 'purchase'
                        GROUP BY c.event_time, c.user_id
                        ;"""

    print("Load customers table - purchase into a df...")
    df = pd.read_sql(load_sql, engine)
    return df


def label_clusters(customers_features_df):
    """Analyse and label clusters"""

    customers_features_df["freq_mean"] = customers_features_df.groupby("clusters")["freq"].transform('mean')
    customers_features_df["spending_mean"] = customers_features_df.groupby("clusters")["spending"].transform('mean')
    customers_features_df["recency_mean"] = customers_features_df.groupby("clusters")["recency"].transform('mean')
    customers_features_df["age_mean"] = customers_features_df.groupby("clusters")["age"].transform('mean')
    
    cluster_labels = []
    for i in range(len(customers_features_df)):
        freq_mean = customers_features_df.loc[i, "freq_mean"]
        spending_mean = customers_features_df.loc[i, "spending_mean"]
        recency_mean = customers_features_df.loc[i, "recency_mean"]
        age_mean = customers_features_df.loc[i, "age_mean"]
    
        if freq_mean < 2 :
            cluster_labels.append("new")
        elif freq_mean < 3 and recency_mean > 49:
            cluster_labels.append("inactive")
        elif freq_mean > 94 and recency_mean < 2:
            cluster_labels.append("vip")
        else:
            cluster_labels.append("regular")

    customers_features_df["cluster_label"] = cluster_labels


def plot_horizontal_bar(customers_features_df):
    """Plot horizontal bar graph"""

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")
    
    cluster_count_df = customers_features_df.groupby("cluster_label")["user_id"].size()
    cluster_count_df = cluster_count_df.reset_index()
    cluster_count_df.columns = ["cluster_label", "total"]
    
    bars = sns.barplot(y="cluster_label", x="total", data=cluster_count_df, orient='h', hue="cluster_label", legend=False, palette="viridis", ax=ax)

    # Add total customer numbers at the end of each bar
    for i, (label, total) in enumerate(zip(cluster_count_df['cluster_label'], cluster_count_df['total'])):
        ax.text(total + total * 0.01,  # x position: slightly right of bar end
                i,                      # y position: same as bar
                f'{total}', # text: formatted number
                va='center',            # vertical alignment
                ha='left'              # horizontal alignment
                #fontsize=10,
                #fontweight='bold'
                )

    # Format Y axis with plain integers, turn off sci notation
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)

    plt.xlabel("Customers")
    plt.ylabel("Clusters")
    plt.grid(True)
    plt.show()


def plot_horizontal_bar2(customers_features_df):
    """Plot horizontal bar graph"""

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    #ax.set_xscale("log")
    
    #print(customers_features_df[customers_features_df["segment2"] == "inactive"])
    #print(customers_features_df.groupby("segment2")["user_id"].count())
    

    cluster_count_df = customers_features_df.groupby("segment2")["user_id"].size()
    cluster_count_df = cluster_count_df.reset_index()
    cluster_count_df.columns = ["segment2", "total"]
    
    bars = sns.barplot(y="segment2", x="total", data=cluster_count_df, orient='h', hue="segment2", legend=False, palette="viridis", ax=ax)

    # Add total customer numbers at the end of each bar
    for i, (label, total) in enumerate(zip(cluster_count_df['segment2'], cluster_count_df['total'])):
        ax.text(total + total * 0.01,  # x position: slightly right of bar end
                i,                      # y position: same as bar
                f'{total}', # text: formatted number
                va='center',            # vertical alignment
                ha='left'              # horizontal alignment
                #fontsize=10,
                #fontweight='bold'
                )

    # Format Y axis with plain integers, turn off sci notation
    #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax.xaxis.get_major_formatter().set_scientific(False)

    plt.xlabel("No. Customers")
    plt.ylabel("Clusters")
    plt.grid(True)
    plt.show()


'''
1) PCA (Principal Component Analysis)
2) PCA is a dimensionality reduction technique.
It takes data with many features (like your 4: freq, spending, recency, age) 
and transforms them into fewer new features (principal components) that:
Capture as much variance as possible (the “information” in the data).
Are uncorrelated with each other (orthogonal).
Are linear combinations of the original features.
3) Visualization → Reduces many dimensions into 2D or 3D, so we can plot it.
Noise reduction → Keeps the most important patterns, drops minor variation.
Feature decorrelation → Makes clustering/ML easier because new features aren’t correlated.
4) describe your data in terms of new axes that summarize the main patterns
5) Think of PCA like finding the best camera angle to take a picture of a 4D object 
— it projects it down to 2D while showing the most meaningful structure.
'''
def plot_clusters(customers_features_df, features_scaled, kmeans):
    """Plot Clusters chart"""

    # Reduce features to 2D for plotting
    pca = PCA(n_components=2)

    pca.fit(features_scaled)
    # Create a DataFrame of loadings
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=["PCA1", "PCA2"], 
        index=['freq', 'spending', 'recency', 'age']
    )
    print(loadings)

    reduced_features = pca.fit_transform(features_scaled)
    reduced_centers = pca.transform(kmeans.cluster_centers_)

    # Scatter plot of customers, colored by cluster
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=customers_features_df["clusters"],
        cmap="tab10",
        alpha=0.7,
        #marker="o",
        s=50
    )

    # Plot centroids in yellow
    plt.scatter(
        reduced_centers[:, 0],
        reduced_centers[:, 1],
        c="yellow",
        s=200,
        edgecolors="black",
        marker="o",
        label="Centroids"
    )

    plt.title("Clusters")
    #plt.xlabel("PCA 1")
    #plt.ylabel("PCA 2")
    #plt.legend(*scatter.legend_elements(), title="Cluster")
    #plt.legend()
    plt.tight_layout()
    plt.show()


def plot_clusters2(customers_features_df, features_scaled, kmeans):
    """Plot Clusters chart"""

    # Reduce features to 2D for plotting
    pca = PCA(n_components=2)

    pca.fit(features_scaled)
    # Create a DataFrame of loadings
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=["PCA1", "PCA2"], 
        index=['freq', 'spending', 'recency', 'age']
    )
    print(loadings)

    reduced_features = pca.fit_transform(features_scaled)
    reduced_centers = pca.transform(kmeans.cluster_centers_)

    # Scatter plot of customers, colored by cluster
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=customers_features_df["clusters"],
        cmap="tab10",
        alpha=0.7,
        #marker="o",
        s=50
    )

    # Plot centroids in yellow
    plt.scatter(
        reduced_centers[:, 0],
        reduced_centers[:, 1],
        c="yellow",
        s=200,
        edgecolors="black",
        marker="o",
        label="Centroids"
    )

    plt.title("Clusters")
    #plt.xlabel("PCA 1")
    #plt.ylabel("PCA 2")
    #plt.legend(*scatter.legend_elements(), title="Cluster")
    #plt.legend()
    plt.tight_layout()
    plt.show()



def plot_bubble_chart(customers_features_df):
    """Plot bubble chart"""

    '''median_freq = customers_features_df.groupby("cluster_label")["freq"].median()
    median_recency = customers_features_df.groupby("cluster_label")["recency"].median()
    avg_spending = customers_features_df.groupby("cluster_label")["spending"].mean()
    count = customers_features_df.groupby("cluster_label")["user_id"].count()

    summary_df = pd.concat([customers_features_df["cluster_label"], median_freq, median_recency, avg_spending, count], axis=1)
    '''
    summary_df = customers_features_df.groupby("cluster_label").agg(
        median_freq=("freq", "median"),
        median_recency=("recency", "median"),
        avg_spending=("spending", "mean"),
        count=("user_id", "count")
    ).reset_index()
    print(summary_df)

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_yscale("log")
    
    sns.scatterplot(
        data=summary_df,
        x="median_recency",
        y="median_freq",
        size="avg_spending",   # bubble size ~ spending
        hue="cluster_label",   # bubble color
        sizes=(100, 2000),     # min/max bubble size
        alpha=0.6,
        palette="Set2"
    )

     # Add labels
    for _, row in summary_df.iterrows():
        ax.text(row["median_recency"]+0.1, row["median_freq"],
                f'Avg "{row["cluster_label"]}": {row["avg_spending"]:.0f}$')

    # Format Y axis with plain integers, turn off sci notation
    #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax.xaxis.get_major_formatter().set_scientific(False)
    # Format Y axis with plain integers, turn off sci notation
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))  # tick every 30 days (~1 month)
    ax.set_xticklabels([int(t/30) for t in ax.get_xticks()])
    ax.legend_.remove()
    #plt.xlabel("Customers")
    #plt.ylabel("Clusters")
    plt.xlabel("Median Recency (month)")
    plt.ylabel("Median Frequency")
    plt.title("Customer Segmentation Bubble Chart")
    #plt.legend(False)
    plt.grid(True)
    plt.show()


def plot_bubble_chart2(customers_features_df):
    """Plot bubble chart"""

    '''median_freq = customers_features_df.groupby("cluster_label")["freq"].median()
    median_recency = customers_features_df.groupby("cluster_label")["recency"].median()
    avg_spending = customers_features_df.groupby("cluster_label")["spending"].mean()
    count = customers_features_df.groupby("cluster_label")["user_id"].count()

    summary_df = pd.concat([customers_features_df["cluster_label"], median_freq, median_recency, avg_spending, count], axis=1)
    '''
    summary_df = customers_features_df.groupby("segment2").agg(
        median_freq=("freq", "median"),
        median_recency=("recency", "median"),
        avg_spending=("spending", "mean"),
        count=("user_id", "count")
    ).reset_index()
    print(summary_df)

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(14, 10))
    #ax.set_yscale("log")
    
    sns.scatterplot(
        data=summary_df,
        x="median_recency",
        y="median_freq",
        size="avg_spending",   # bubble size ~ spending
        hue="segment2",   # bubble color
        sizes=(100, 2000),     # min/max bubble size
        alpha=0.6,
        #palette="Set2"
        palette="viridis"
    )

     # Add labels
    for _, row in summary_df.iterrows():
        ax.text(row["median_recency"]+1, row["median_freq"],
                f'Avg "{row["segment2"]}": {row["avg_spending"]:.0f}$')

    # Format Y axis with plain integers, turn off sci notation
    #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax.xaxis.get_major_formatter().set_scientific(False)
    # Format Y axis with plain integers, turn off sci notation
    
    #ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax.yaxis.get_major_formatter().set_scientific(False)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))  # tick every 30 days (~1 month)
    
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([int(t/30) for t in ax.get_xticks()])
    ax.legend_.remove()
 
    plt.xlabel("Median Recency (month)")
    plt.ylabel("Median Frequency")
    plt.title("Customer Segmentation Bubble Chart")
    plt.grid(True)
    plt.show()


def cluster_customers(customers_df):
    """"""

    freq = customers_df.groupby("user_id")["basket_price"].count()
    spending = customers_df.groupby("user_id")["basket_price"].sum()

    ref_date = pd.to_datetime("2023-03-01")
    
    last_purchase_dates = customers_df.groupby("user_id")["event_time"].max().dt.normalize() #remove time component
    recency = (ref_date - last_purchase_dates).dt.days

    first_purchase_date = customers_df.groupby("user_id")["event_time"].min().dt.normalize()
    age = (ref_date - first_purchase_date).dt.days

    customers_features_df = pd.concat([freq, spending, recency, age], axis=1)    
    customers_features_df = customers_features_df.reset_index()
    customers_features_df.columns = ["user_id", "freq", "spending", "recency", "age"]

    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(customers_features_df[['freq', 'spending', "recency", "age"]])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    customers_features_df["clusters"] = kmeans.fit_predict(features_scaled)

    # Calculate silhouette score
    #silhouette_avg = silhouette_score(features_scaled, customers_features_df['clusters'])
    #print(f"Silhouette Score: {silhouette_avg:.3f}")
    
    label_clusters(customers_features_df)
    print(customers_features_df)

    #1
    plot_horizontal_bar(customers_features_df)
    #2
    plot_bubble_chart(customers_features_df)
    #3
    plot_clusters(customers_features_df, features_scaled, kmeans)
    

'''
1) np.log1p(x) is the natural logarithm of (1 + x)  # log(1 + x)
Reduces skew in data
Many customer features like recency, spending, freq are heavily skewed.
For example, most customers might have freq = 1–2 but a few have freq = 50.
Taking log1p compresses large values while keeping small values relatively separate.
Keeps zeros safe
Regular np.log(x) fails for x = 0.
np.log1p(0) = log(1) = 0, so it handles zero naturally.
'''
def cluster_customers_rf(customers_df):
    """Cluster by recency and frequency"""

    freq = customers_df.groupby("user_id")["basket_price"].count()
    spending = customers_df.groupby("user_id")["basket_price"].sum()

    ref_date = pd.to_datetime("2023-03-01")
    
    last_purchase_dates = customers_df.groupby("user_id")["event_time"].max().dt.normalize() #remove time component
    recency = (ref_date - last_purchase_dates).dt.days

    customers_features_df = pd.concat([freq, spending, recency], axis=1)
    customers_features_df = customers_features_df.reset_index()
    customers_features_df.columns = ["user_id", "freq", "spending", "recency"]

    features = customers_features_df[['freq', 'recency']].copy()
    #print(features.info())
    features = features.apply(lambda x: np.log1p(x))

    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customers_features_df["cluster_rf"] = kmeans.fit_predict(features_scaled)

    # Look at cluster centers
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=['freq', 'recency']
    )
    centroids['cluster'] = centroids.index
    
    sorted_clusters = centroids.sort_values('recency')
    print(sorted_clusters.iloc[0])

    cluster_mapping = {
        sorted_clusters.iloc[0]['cluster']: "new",
        sorted_clusters.iloc[1]['cluster']: "regular",
        sorted_clusters.iloc[2]['cluster']: "inactive"
    }
    customers_features_df["segment1"] = customers_features_df["cluster_rf"].map(cluster_mapping)

    #  For customers in regular segment1, get spending data and do Kmeans clustering to get high and low spending.
    scaler2 = RobustScaler()
    #scaler2 = MinMaxScaler()
    

    regular_mask = customers_features_df['segment1'] == "regular"
    regular_data = customers_features_df.loc[regular_mask, ['spending']]

    #features = customers_features_df[['freq', 'recency']].copy()
    #features = features.apply(lambda x: np.log1p(x))
    #features2 = customers_features_df.loc[regular_mask, ['spending']].copy()
    #print(type(features2))
    #print(features2.info())
    #print("-------")
    #print(features2.isna().sum())

    #features2 = features2.apply(lambda x: np.log1p(x))
    regular_scaled = scaler2.fit_transform(regular_data)

    # Split into High vs Low spending
    kmeans_spend = KMeans(n_clusters=2, random_state=42)
    customers_features_df.loc[regular_mask, 'cluster_spending'] = kmeans_spend.fit_predict(regular_scaled)
    print(customers_features_df)

    spend_centroids = pd.DataFrame(
        scaler2.inverse_transform(kmeans_spend.cluster_centers_),
        columns=['spending']
    )
    gold_cluster_id = spend_centroids['spending'].idxmax()
    print(gold_cluster_id)
    print(spend_centroids)
    #print(customers_features_df)

    # Assign Gold vs Regular
    #df.loc[regular_mask, 'segment'] = df.loc[regular_mask, 'value_cluster'].apply(
    #    lambda x: "Gold Customer" if x == gold_cluster_id else "Regular Customer"
    #)

    customers_features_df.loc[regular_mask, "segment2"] = customers_features_df.loc[regular_mask, 'cluster_spending'].apply(
        lambda x: "vip" if x == gold_cluster_id else "regular"
    )

    customers_features_df.loc[~regular_mask, "segment2"] = customers_features_df.loc[~regular_mask, 'segment1']

    print(customers_features_df)

    plot_horizontal_bar2(customers_features_df)
    plot_bubble_chart2(customers_features_df)
    plot_clusters2(customers_features_df)


def main():
    """main()"""

    try:
        engine = get_db_conn()

        schema = "customer2"
        table = "customers"
        
        customers_df = load_df_sql(schema, table, engine)

        print(customers_df)
        #cluster_customers(customers_df)
        cluster_customers_rf(customers_df)



    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
