import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np


def get_db_conn():
    """Establishes and returns a connection to the PostgreSQL database."""
    
    engine = create_engine("postgresql://ssian:mysecretpassword@localhost:5432/piscineds")

    return engine


#Keep only the "purchase" data of "event_type" column
def load_df_sql(schema, table, engine):
    """Load customers group by event_type into a df, return the df"""

    load_sql = f"""SELECT c.event_time, c.user_id, sum(c.price) as sum_price FROM {schema}.{table} c
                        WHERE c.event_type = 'purchase'
                        GROUP BY c.event_time, c.user_id
                        ;"""

    print("Load customers table - purchase into a df...")
    df = pd.read_sql(load_sql, engine)
    return df


def plot_histogram_customer_freq(customers_df):
    """Plot a histogram chart of customers vs frequency"""

    customers_groupby_df = customers_df.groupby("user_id").size()
    customers_groupby_df = customers_groupby_df.reset_index()
    customers_groupby_df.columns = ["user_id", "freq"]

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_yscale("log")

    max_freq = customers_groupby_df["freq"].max()
    #bins=[0, 10, 20, 30....]
    bins = np.arange(0, max_freq + 10, 10)
    #bins = np.arange(0, 50, 10)

    sns.histplot(customers_groupby_df["freq"], bins=bins)

    # Format Y axis with plain integers, turn off sci notation
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    plt.xticks(bins)
    plt.xlabel("Frequency")
    plt.ylabel("Customers")
    plt.title("Customers Frequency")
    plt.grid(True)
    plt.show()


def plot_histogram_customer_spending(customers_df):
    """Plot a histogram chart of customers spending $"""

    customers_groupby_df = customers_df.groupby("user_id")["sum_price"].sum()
    customers_groupby_df = customers_groupby_df.reset_index()
    customers_groupby_df.columns = ["user_id", "spending"]

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_yscale("log")

    max_spending = customers_groupby_df["spending"].max()
    #bins=[0, 10, 20, 30....]
    bins = np.arange(0, max_spending + 200, 200)
    #bins = np.arange(0, 300, 50)

    sns.histplot(customers_groupby_df["spending"], bins=bins)

    # Format Y axis with plain integers, turn off sci notation
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    plt.xticks(bins)
    plt.xlabel("Monetary value in $")
    plt.ylabel("Customers")
    plt.title("Customers Spending")
    plt.grid(True)
    plt.show()


def main():
    """main()"""

    try:
        engine = get_db_conn()

        schema = "customer2"
        table = "customers"
        customers_df = load_df_sql(schema, table, engine)

        plot_histogram_customer_freq(customers_df)
        plot_histogram_customer_spending(customers_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
