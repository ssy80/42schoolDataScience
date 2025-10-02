import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import seaborn as sns


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


# daily_counts = event_time, total customers
# multi plot: fig, (ax1, ax2) = plt.subplots(1, 2)) -> 1 row, 2 plot
def plot_purchases(purchases_df):
    """Plot line graph of total customers per day"""

    daily_counts = purchases_df.groupby(purchases_df['event_time'].dt.date).size() #group by date - 2022-10-01
    daily_counts_df = daily_counts.reset_index()
    daily_counts_df.columns = ["event_time", "count"]

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x="event_time", y="count", data=daily_counts_df)
    
    # Set ticks to every month %b - "OCT", "NOV"
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    min_month = daily_counts_df["event_time"].min()
    max_month = daily_counts_df["event_time"].max()
    plt.xlim(min_month, max_month)
    
    plt.ylabel("Number of Customers")
    plt.grid(True)
    plt.show()


# add a column purchases_df['month'] to_period = 2022-10, 2022-11
# to_timestamp() - convert 2022-10 to 2011-10-01 (proper datetime)
def plot_sales(purchases_df):
    """Plot bar graph of sales per month"""

    purchases_df['month'] = purchases_df['event_time'].dt.to_period('M')
    monthly_sales = purchases_df.groupby('month')['sum_price'].sum()
    monthly_sales.index = monthly_sales.index.to_timestamp()

    monthly_sales_df = monthly_sales.reset_index()
    monthly_sales_df.columns = ["month", "sales"]

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(monthly_sales_df["month"], monthly_sales_df["sales"], width=20)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    plt.ylabel("Total sales in million of $")
    plt.xlabel("Month")
    plt.grid(True)
    plt.show()


def plot_average_sales(customers_df):
    """Plot average sales per customer"""

    customers_daily = customers_df.groupby(customers_df['event_time'].dt.date).size()
    sales_daily = customers_df.groupby(customers_df['event_time'].dt.date)['sum_price'].sum()
    
    customers_sales_df = customers_daily.reset_index()
    customers_sales_df.columns = ["event_time", "num_customers"]
    customers_sales_df["sales"] = sales_daily.values
    customers_sales_df["sales_per_customer"] = customers_sales_df["sales"].div(customers_sales_df["num_customers"])

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(customers_sales_df["event_time"] , customers_sales_df["sales_per_customer"])
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    min_month = customers_sales_df["event_time"].min()
    max_month = customers_sales_df["event_time"].max()
    
    plt.xlim(min_month, max_month)
    plt.ylabel("Average spend/customers in $")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """main()"""

    try:
        engine = get_db_conn()

        schema = "customer2"
        table = "customers"
        customers_df = load_df_sql(schema, table, engine)

        plot_purchases(customers_df)
        plot_sales(customers_df)
        plot_average_sales(customers_df)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
