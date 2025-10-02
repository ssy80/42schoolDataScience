import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


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


def plot_horizontal_bar(customers_features_df):
    """Plot horizontal bar graph"""

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    cluster_count_df = customers_features_df.groupby("segment2")["user_id"].size()
    cluster_count_df = cluster_count_df.reset_index()
    cluster_count_df.columns = ["segment2", "total"]
    
    sns.barplot(y="segment2", x="total", data=cluster_count_df, orient='h', hue="segment2", legend=False, palette="viridis", ax=ax)

    # Add total customer numbers at the end of each bar
    for i, (label, total) in enumerate(zip(cluster_count_df['segment2'], cluster_count_df['total'])):
        ax.text(total + total * 0.01, i, f'{total}', va='center', ha='left')

    plt.xlabel("No. Customers")
    plt.ylabel("Clusters")
    plt.grid(True)
    plt.show()


def plot_bubble_chart(customers_features_df):
    """Plot bubble chart"""

    summary_df = customers_features_df.groupby("segment2").agg(
        median_freq=("freq", "median"),
        median_recency=("recency", "median"),
        avg_spending=("spending", "mean"),
        count=("user_id", "count")
    ).reset_index()

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.scatterplot(
        data=summary_df,
        x="median_recency",
        y="median_freq",
        size="avg_spending",
        hue="segment2",
        sizes=(100, 2000),
        alpha=0.6,
        palette="viridis"
    )

    # Add labels to bubble
    for _, row in summary_df.iterrows():
        ax.text(row["median_recency"]+1, row["median_freq"],
                f'Avg "{row["segment2"]}": {row["avg_spending"]:.0f}$')

    # tick every 30 days (~1 month)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))  
    
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([int(t/30) for t in ax.get_xticks()])
    ax.legend_.remove()
 
    plt.xlabel("Median Recency (month)")
    plt.ylabel("Median Frequency")
    plt.title("Customer Segmentation Bubble Chart")
    plt.grid(True)
    plt.show()


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
def cluster_customers(customers_df):
    """Kmeans cluster by recency, frequency 1st into new, regular, inactive
    2nd Kmeans cluster regular segment spending into high, low spending.
    """

    freq = customers_df.groupby("user_id")["basket_price"].count()
    spending = customers_df.groupby("user_id")["basket_price"].sum()

    ref_date = pd.to_datetime("2023-03-01")
    last_purchase_dates = customers_df.groupby("user_id")["event_time"].max().dt.normalize() #remove time component
    recency = (ref_date - last_purchase_dates).dt.days

    customers_features_df = pd.concat([freq, spending, recency], axis=1)
    customers_features_df = customers_features_df.reset_index()
    customers_features_df.columns = ["user_id", "freq", "spending", "recency"]

    scaler = RobustScaler()
    features = customers_features_df[['freq', 'recency']].copy()
    features = features.apply(lambda x: np.log1p(x))
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customers_features_df["cluster_rf"] = kmeans.fit_predict(features_scaled)

    # check centroids for segmentation to new, regular, inactive
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=['freq', 'recency']
    )
    centroids['cluster'] = centroids.index
    
    sorted_clusters = centroids.sort_values('recency')
    cluster_mapping = {
        sorted_clusters.iloc[0]['cluster']: "new",
        sorted_clusters.iloc[1]['cluster']: "regular",
        sorted_clusters.iloc[2]['cluster']: "inactive"
    }
    customers_features_df["segment1"] = customers_features_df["cluster_rf"].map(cluster_mapping)

    #  For customers in regular segment1, get spending data and do Kmeans clustering to get high and low spending.
    scaler2 = RobustScaler()
    regular_mask = customers_features_df['segment1'] == "regular"
    regular_data = customers_features_df.loc[regular_mask, ['spending']]
    regular_scaled = scaler2.fit_transform(regular_data)

    # Split into High vs Low spending
    kmeans_spend = KMeans(n_clusters=2, random_state=42)
    customers_features_df.loc[regular_mask, 'cluster_spending'] = kmeans_spend.fit_predict(regular_scaled)

    # check spending_centroids for segmentation to high, low spender
    spending_centroids = pd.DataFrame(
        scaler2.inverse_transform(kmeans_spend.cluster_centers_),
        columns=['spending']
    )
    vip_cluster_id = spending_centroids['spending'].idxmax()

    customers_features_df.loc[regular_mask, "segment2"] = customers_features_df.loc[regular_mask, 'cluster_spending'].apply(
        lambda x: "vip" if x == vip_cluster_id else "regular"
    )

    customers_features_df.loc[~regular_mask, "segment2"] = customers_features_df.loc[~regular_mask, 'segment1']

    plot_horizontal_bar(customers_features_df)
    plot_bubble_chart(customers_features_df)


def main():
    """main()"""

    try:
        engine = get_db_conn()

        schema = "customer2"
        table = "customers"
        customers_df = load_df_sql(schema, table, engine)

        cluster_customers(customers_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
