

def get_insert_spread_sql(table_name: str) -> str:
    return f"""
    INSERT INTO {table_name} (
        text, source, class
    )
    VALUES (?, ?, ?)
    """


def get_select_all_sql(table_name: str) -> str:
    return f"SELECT * FROM {table_name}"



def get_info_table_sql(table_name: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        class_news TEXT NOT NULL
    )
    """


def get_update_info_sql(table_name: str) -> str:
    return f"""
    UPDATE {table_name} 
    SET text = :text, source = :source, class_news := class_news
    """

def get_insert_info_sql(table_name: str) -> str:
    return f"""
    INSERT INTO {table_name} (text, source, class_news)
    VALUES (:text, :source, :class_news)
    """

def get_select_column_sql(table_name: str, column_name: str) -> str:
    return f"SELECT {column_name} FROM {table_name}"