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


'''def union_tables(cur, schema, new_table, tables):
    """Creates a new table by performing a UNION of a list of tables."""

    union_tables_sql = f"CREATE TABLE {schema}.{new_table} AS"

    for i in range(len(tables)):
        if i == 0:
            union_tables_sql += f" SELECT * FROM {schema}.{tables[i]}"
        else:
            union_tables_sql += f" UNION ALL SELECT * FROM {schema}.{tables[i]}"
    
    cur.execute(union_tables_sql)'''
def remove_duplicates(cur, schema, table):
    


def main():
    """main()"""

    try:
        conn = get_db_conn()
        cur = conn.cursor()

        schema = "customer2"
        table = "customers"
        #tables = ["data_2022_oct", "data_2022_nov", "data_2022_dec", "data_2023_jan", "data_2023_feb"]
        remove_duplicates(cur, schema, table)

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

"""
DELETE FROM customer2.customers 
WHERE ctid IN (
    SELECT c1.ctid
    FROM customer2.customers c1
    INNER JOIN customer2.customers c2 ON 
        c1.event_type = c2.event_type 
        AND c1.product_id = c2.product_id
        AND c1.user_id = c2.user_id
        AND c1.user_session = c2.user_session
        AND c1.price = c2.price
        AND ABS(EXTRACT(EPOCH FROM (c1.event_time - c2.event_time))) <= 1
		OR c1.event_time = c2.event_time
        AND c1.ctid > c2.ctid  -- Keep the earliest record
);
"""