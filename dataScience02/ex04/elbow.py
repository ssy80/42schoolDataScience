import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler


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


'''
1) Calculate the rate of decrease
2) Find the first point where it drop less than 40%
3) Default clusters 3 if no clear elbow, or len(k_range) if < 3
4) Elbow Method shows diminishing returns beyond optimal k
'''
def get_elbow_point(sse, k_range):
    """Find the elbow point where SSE decrease slows down"""
    
    decreases = []
    for i in range(1, len(sse)):
        decrease = sse[i-1] - sse[i]
        decreases.append(decrease)
    
    for i in range(1, len(decreases)):
        if decreases[i] < decreases[i-1] * 0.6:
            return k_range[i]
    
    if len(k_range) < 3:
        return len(k_range)
    else:
        return 3


'''
1) SSE stands for Sum of Squared Errors (also called Within-Cluster Sum of Squares or Inertia). 
It's a key metric in K-means clustering that measures how compact and well-separated your clusters are.
SSE calculates the total squared distance between each data point and its assigned cluster center (centroid)
(sse = kmeans.inertia_), SSE measures how "spread out" or "compact" your clusters are.
    Low SSE = Tight, compact clusters (good!)
    High SSE = Spread out, loose clusters (bad!)
2) n_init=10 (runs 10 times, keeps best)
3) Use RobustScaler (better for outliers!)
4) K-means uses Euclidean distance in multi-dimensional space to measure similarity between customers.
'''
def plot_elbow(customers_df):
    """Plot elbow graph to get best K value"""

    freq = customers_df.groupby("user_id")["basket_price"].count()
    spending = customers_df.groupby("user_id")["basket_price"].sum()

    ref_date = pd.to_datetime("2023-03-01")
    last_purchase_dates = customers_df.groupby("user_id")["event_time"].max().dt.normalize() #remove time component
    recency = (ref_date - last_purchase_dates).dt.days

    customers_features_df = pd.concat([freq, spending, recency], axis=1)
    customers_features_df = customers_features_df.reset_index()
    customers_features_df.columns = ["user_id", "freq", "spending", "recency"]

    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(customers_features_df[["freq", "recency", "spending"]])

    sse = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        sse.append(kmeans.inertia_)

    optimal_clusters = get_elbow_point(sse, k_range)
    print(f"Optimal No. of clusters is {optimal_clusters}")

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse)
    plt.xlabel("No. of clusters")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.title(f"The Elbow Method - Optimal No. of clusters is {optimal_clusters}")
    plt.grid(True)
    plt.show()


def main():
    """main()"""

    try:
        engine = get_db_conn()

        schema = "customer2"
        table = "customers"
        customers_df = load_df_sql(schema, table, engine)

        plot_elbow(customers_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
