"""
ULTRON Database Module
SQLite operations for storing authorized face embeddings.
"""

import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).parent / "ultron.db"


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS authorized_users (
            user_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()


def add_user(user_id: str, embedding: np.ndarray) -> bool:
    """
    Add a new authorized user with their face embedding.
    
    Args:
        user_id: Unique identifier for the user
        embedding: Face embedding as numpy array
        
    Returns:
        True if successful, False if user already exists
    """
    if user_exists(user_id):
        return False
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert numpy array to bytes for storage
    embedding_bytes = embedding.astype(np.float32).tobytes()
    created_at = datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO authorized_users (user_id, embedding, created_at) VALUES (?, ?, ?)",
        (user_id, embedding_bytes, created_at)
    )
    
    conn.commit()
    conn.close()
    return True


def get_all_users() -> list:
    """
    Retrieve all authorized users and their embeddings.
    
    Returns:
        List of tuples: (user_id, embedding_array, created_at)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT user_id, embedding, created_at FROM authorized_users")
    rows = cursor.fetchall()
    
    conn.close()
    
    # Convert bytes back to numpy arrays
    users = []
    for user_id, embedding_bytes, created_at in rows:
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        users.append((user_id, embedding, created_at))
    
    return users


def user_exists(user_id: str) -> bool:
    """Check if a user ID already exists in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT 1 FROM authorized_users WHERE user_id = ?", (user_id,))
    exists = cursor.fetchone() is not None
    
    conn.close()
    return exists


def count_users() -> int:
    """Return the number of registered users."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM authorized_users")
    count = cursor.fetchone()[0]
    
    conn.close()
    return count


def delete_user(user_id: str) -> bool:
    """Delete a user from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM authorized_users WHERE user_id = ?", (user_id,))
    deleted = cursor.rowcount > 0
    
    conn.commit()
    conn.close()
    return deleted
