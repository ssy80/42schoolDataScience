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
    """Load customers purchase data into a df, return the df"""

    load_sql = f"""SELECT c.price FROM {schema}.{table} c
                        WHERE c.event_type = 'purchase'
                        ;"""

    print("Load customers table - purchase into a df...")
    df = pd.read_sql(load_sql, engine)
    return df


def plot_boxplot(customers_df):
    """Plot a boxplot of purchase items"""

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=customers_df["price"])

    plt.xlabel("Price")
    plt.grid(True)
    plt.show()


def plot_boxplot_zoom(customers_df):
    """Plot a boxplot of purchase items without outliers from price -1 to 13"""

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=customers_df["price"], showfliers=False)

    plt.xlim(-1, 13)
    plt.xlabel("Price")
    plt.grid(True)
    plt.show()


def load_basket_sql(schema, table, engine):
    """Load customers group by event_type into a df, return the df"""

    load_sql = f"""SELECT c.event_time, c.user_id, sum(c.price) as sum_price FROM {schema}.{table} c
                        WHERE c.event_type = 'purchase'
                        GROUP BY c.event_time, c.user_id
                        ;"""

    print("Load customers table - purchase into a df...")
    df = pd.read_sql(load_sql, engine)
    return df


def plot_boxplot_avg(customers_basket_df):
    """Plot boxplot for average basket per user"""

    avg_basket_df = customers_basket_df.groupby("user_id")["sum_price"].mean()
    avg_basket_df = avg_basket_df.reset_index()
    avg_basket_df.columns = ["user_id", "avg_basket"]

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=avg_basket_df["avg_basket"])

    plt.xlim(-15, 105)
    plt.grid(True)
    plt.show()


def main():
    """main()"""

    try:
        engine = get_db_conn()

        schema = "customer2"
        table = "customers"
        
        customers_df = load_df_sql(schema, table, engine)
        
        pd.set_option("display.float_format", "{:.6f}".format)
        print(customers_df.describe())

        plot_boxplot(customers_df)
        plot_boxplot_zoom(customers_df)

        customer_basket_df = load_basket_sql(schema, table, engine)
        plot_boxplot_avg(customer_basket_df)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
