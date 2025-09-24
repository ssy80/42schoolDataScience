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


def alter_customers_table(cur, schema, table):
    """Alter customers table to include items columns"""

    alter_sql = sql.SQL("""ALTER TABLE {schema}.{table}
                            ADD COLUMN category_id BIGINT,
                            ADD COLUMN category_code TEXT,
                            ADD COLUMN brand TEXT
                        ;""").format(schema=sql.Identifier(schema), table=sql.Identifier(table))
    print("Alter customers table...")
    cur.execute(alter_sql)


def remove_items_duplicates(cur, schema, table):
    """remove duplicates rows in items table"""

    remove_duplicates_sql = sql.SQL("""DELETE FROM {schema}.{table}
                                    WHERE ctid IN (
                                    SELECT i1.ctid
                                    FROM items.items i1
                                    INNER JOIN items.items i2 ON 
                                        i1.product_id = i2.product_id
                                        AND i1.ctid > i2.ctid
                                        WHERE i1.category_id is null
                                        AND i1.category_code is null
                                        AND i1.brand is null
                                );""").format(schema=sql.Identifier(schema), table=sql.Identifier(table))
    print("Removing items duplicates now...")
    cur.execute(remove_duplicates_sql)


def update_customers_items(cur, customers_schema, customers_table, items_schema, items_table):
    update_sql = sql.SQL("""UPDATE {customers_schema}.{customers_table} c
                                SET category_id = i.category_id,
                                category_code = i.category_code,
                                brand = i.brand
                                FROM {items_schema}.{items_table} i
                                WHERE c.product_id = i.product_id
                            ;""").format(
                                customers_schema=sql.Identifier(customers_schema),
                                customers_table=sql.Identifier(customers_table),
                                items_schema=sql.Identifier(items_schema),
                                items_table=sql.Identifier(items_table),
                                )
    print("Update customers items now...")
    cur.execute(update_sql)


def main():
    """main()"""

    try:
        conn = get_db_conn()
        cur = conn.cursor()

        customers_schema = "customer2"
        customers_table = "customers"
        items_schema = "items"
        items_table = "items"

        alter_customers_table(cur, customers_schema, customers_table)
        remove_items_duplicates(cur, items_schema, items_table)
        update_customers_items(cur, customers_schema, customers_table, items_schema, items_table)

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
