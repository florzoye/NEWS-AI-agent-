import sqlite3
from typing import List, Dict
from db.schemas import (
    get_info_table_sql,
    get_update_info_sql,
    get_insert_info_sql,
    get_select_all_sql
)

class NewsSQL:
    def __init__(self, cursor: sqlite3.Cursor):
        self.cursor = cursor

    def create_table(self, table_name: str) -> None:
        self.cursor.execute(get_info_table_sql(table_name))

    def add_info(self, table_name: str, news: Dict) -> None:
        existing = self.cursor.fetchone()
        
        if existing:
            self.cursor.execute(
                get_update_info_sql(table_name),
                news
            )
        else:
            self.cursor.execute(
                get_insert_info_sql(table_name),
                news
            )
            
    
    def get_all(self, table_name: str) -> List[Dict]:
        self.cursor.execute(get_select_all_sql(table_name))
        columns = [col[0] for col in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def clear_table(self, table_name):
        self.cursor.execute(f"DELETE FROM {table_name}")