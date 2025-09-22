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


def get_csv_files(dir_path):

    files = os.listdir(dir_path)

    csv_files = []
    for f in files:
        if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith('.csv'):
            csv_files.append(f)
    return csv_files


def create_tables(dir_path, csv_files, cur, schema):

    tables = [os.path.splitext(f)[0] for f in csv_files]
    print(tables)
    for t in tables:
        create_table_sql = sql.SQL("""CREATE TABLE {sch}.{table}(
                    event_time TIMESTAMPTZ,
                    event_type TEXT,
                    product_id INTEGER,
                    price NUMERIC(10,2),
                    user_id BIGINT,
                    user_session UUID
                    );""").format(sch=sql.Identifier(schema), table=sql.Identifier(t))
        cur.execute(create_table_sql)


def create_schema(schema, cur):
    create_schema_sql = sql.SQL("""CREATE SCHEMA {sch};""").format(sch=sql.Identifier(schema))
    cur.execute(create_schema_sql)


# use copy for bulk loading, insert too slow
def load_tables(dir_path, csv_files, cur, schema):

    for f in csv_files:
        table = os.path.splitext(f)[0]
        table_name = f"{schema}.{table}"
        filepath = os.path.join(dir_path, f)

        with open(filepath, "r", encoding="utf-8") as csv_file:
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
            cur.copy_expert(sql, csv_file)


def main():
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        dir_path = "./customer"
        schema = "customer3"
        csv_files = get_csv_files(dir_path)

        create_schema(schema, cur)
        create_tables(dir_path, csv_files, cur, schema)
        print("Loading data from .csv")
        load_tables(dir_path, csv_files, cur, schema)

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
