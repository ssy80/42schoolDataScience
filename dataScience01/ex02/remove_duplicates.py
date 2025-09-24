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


# Deletes rows based on their physical row ID (ctid) — a system column in PostgreSQL 
# that uniquely identifies a row version inside a table.
def remove_duplicates(cur, schema, table):
    """Remove duplicates rows in table"""

    remove_duplicates_sql = sql.SQL("""DELETE FROM {schema}.{table}
                                WHERE ctid IN (
                                    SELECT c1.ctid
                                    FROM {schema}.{table} c1
                                    INNER JOIN {schema}.{table} c2 ON 
                                        c1.event_type = c2.event_type 
                                        AND c1.product_id = c2.product_id
                                        AND c1.user_id = c2.user_id
                                        AND c1.user_session = c2.user_session
                                        AND c1.price = c2.price
                                        AND c1.event_time BETWEEN c2.event_time - INTERVAL '1 second'
                                                            AND c2.event_time + INTERVAL '1 second'
                                        AND c1.ctid > c2.ctid
                                );""").format(schema=sql.Identifier(schema), table=sql.Identifier(table))
    print("Removing duplicates now...")
    cur.execute(remove_duplicates_sql)


def main():
    """main()"""

    try:
        conn = get_db_conn()
        cur = conn.cursor()

        schema = "customer2"
        table = "customers"
        remove_duplicates(cur, schema, table)

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
