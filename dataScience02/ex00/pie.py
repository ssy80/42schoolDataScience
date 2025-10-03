import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def get_db_conn():
    """Establishes and returns a connection to the PostgreSQL database."""
    
    engine = create_engine("postgresql://ssian:mysecretpassword@localhost:5432/piscineds")

    return engine


def load_df_sql(schema, table, engine):
    """Load customers group by event_type into a df, return the df"""

    load_sql = f"""SELECT event_type, count(*) as count 
                        FROM {schema}.{table}
                        GROUP BY event_type
                        ;"""

    print("Load customers table into a df...")
    df = pd.read_sql(load_sql, engine)
    return df


# %1.1f%%" = 23.1%
def plot_pie(customers_df):
    """Plot a pie chart"""

    plt.pie(
        customers_df["count"],
        labels=customers_df["event_type"],
        autopct="%1.1f%%",
        startangle=180,
    )

    plt.title("Event Type Distribution")
    plt.show()


def main():
    """main()"""

    try:
        engine = get_db_conn()

        schema = "customer2"
        table = "customers"
        customers_df = load_df_sql(schema, table, engine)
        plot_pie(customers_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
