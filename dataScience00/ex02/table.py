import os
from psycopg2 import sql, connect


def get_db_conn():

    conn = connect(
    host="localhost",
    port=5432,
    database="piscineds",
    user="ssian",
    password="mysecretpassword"
    )
    return conn


def create_table(table, cur, schema):

    create_table_sql = sql.SQL("""CREATE TABLE {sch}.{table}(
                event_time TIMESTAMPTZ,
                event_type TEXT,
                product_id INTEGER,
                price NUMERIC(10,2),
                user_id BIGINT,
                user_session UUID
                );""").format(sch=sql.Identifier(schema), table=sql.Identifier(table))
    cur.execute(create_table_sql)


def create_schema(schema, cur):
    create_schema_sql = sql.SQL("""CREATE SCHEMA {sch};""").format(sch=sql.Identifier(schema))
    cur.execute(create_schema_sql)


# use copy for bulk loading, insert too slow
def load_table(filepath, cur, schema, table):

    table_name = f"{schema}.{table}"

    with open(filepath, "r", encoding="utf-8") as f:
        sql = f"""
            COPY {table_name} (
                    event_time,
                    event_type,
                    product_id,
                    price,
                    user_id,
                    user_session
                )
            FROM STDIN WITH CSV HEADER;
        """
        cur.copy_expert(sql, f)


def main():
    try:
        conn = get_db_conn()
        cur = conn.cursor()

        schema = "customer4"
        table = "data_2022_dec"
        filepath = "./customer/data_2022_dec.csv"

        create_schema(schema, cur)
        create_table(table, cur, schema)
        print("Loading data from .csv")
        load_table(filepath, cur, schema, table)
    
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
