from psycopg2 import sql, connect


def get_db_conn():
    """Establishes and returns a connection to the PostgreSQL database."""

    conn = connect(
        host="localhost",
        port=5432,
        database="piscineds",
        user="ssian",
        password="mysecretpassword"
    )
    return conn

def union_tables(cur, schema, new_table, tables):
    """Creates a new table by performing a UNION of a list of tables."""

    union_tables_sql = f"CREATE TABLE {schema}.{new_table} AS"

    for i in range(len(tables)):
        if i == 0:
            union_tables_sql += f" SELECT * FROM {schema}.{tables[i]}"
        else:
            union_tables_sql += f" UNION ALL SELECT * FROM {schema}.{tables[i]}"
    
#    print(union_tables_sql)
    cur.execute(union_tables_sql)

def main():
    """main()"""

    try:
        conn = get_db_conn()
        cur = conn.cursor()

        schema = "customer2"
        new_table = "customers"
        #table = "data_2023_feb"
        #filepath = "./customer/data_2023_feb.csv"

        #create_schema(schema, cur)
        #create_table(table, cur, schema)
        #print("Loading data from .csv")
        #load_table(filepath, cur, schema, table)
        tables = ["data_2022_oct", "data_2022_nov", "data_2022_dec", "data_2023_jan", "data_2023_feb"]
        union_tables(cur, schema, new_table, tables)

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()