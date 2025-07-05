import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sqlite3
import threading

logger = logging.getLogger(__name__)

class DatabaseHandler:
    def __init__(self):
        self.db_path = Path("/app/user_data/tts_database.db")
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize database tables"""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Synthesis history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS synthesis_history (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        audio_path TEXT NOT NULL,
                        settings TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        duration REAL,
                        model_name TEXT,
                        speaker TEXT,
                        language TEXT
                    )
                ''')
                
                # Voice profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS voice_profiles (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        file_path TEXT NOT NULL,
                        features TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        quality INTEGER,
                        sample_rate INTEGER,
                        duration REAL
                    )
                ''')
                
                # User preferences table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Model usage statistics
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        usage_count INTEGER DEFAULT 1,
                        last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_duration REAL DEFAULT 0,
                        average_processing_time REAL DEFAULT 0
                    )
                ''')
                
                conn.commit()
                conn.close()
                
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    def save_synthesis(
        self, 
        request_id: str,
        text: str,
        audio_path: str,
        settings: Dict[str, Any]
    ):
        """Save synthesis record to database"""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO synthesis_history 
                    (id, text, audio_path, settings, model_name, speaker, language)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    request_id,
                    text,
                    audio_path,
                    json.dumps(settings),
                    settings.get("model_name", ""),
                    settings.get("speaker", ""),
                    settings.get("language", "en")
                ))
                
                # Update model usage statistics
                model_name = settings.get("model_name", "default")
                cursor.execute('''
                    INSERT INTO model_usage (model_name)
                    VALUES (?)
                    ON CONFLICT(model_name) DO UPDATE SET
                        usage_count = usage_count + 1,
                        last_used = CURRENT_TIMESTAMP
                ''', (model_name,))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error saving synthesis: {str(e)}")
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get synthesis history"""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, text, audio_path, settings, created_at, 
                           model_name, speaker, language
                    FROM synthesis_history
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                conn.close()
                
                history = []
                for row in rows:
                    history.append({
                        "id": row[0],
                        "text": row[1],
                        "audio_path": row[2],
                        "settings": json.loads(row[3]),
                        "created_at": row[4],
                        "model_name": row[5],
                        "speaker": row[6],
                        "language": row[7]
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Error getting history: {str(e)}")
            return []
    
    def save_user_preference(self, key: str, value: Any):
        """Save user preference"""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO user_preferences (key, value)
                    VALUES (?, ?)
                ''', (key, json.dumps(value)))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error saving preference: {str(e)}")
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference"""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT value FROM user_preferences WHERE key = ?
                ''', (key,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return json.loads(row[0])
                return default
                
        except Exception as e:
            logger.error(f"Error getting preference: {str(e)}")
            return default
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get model usage statistics"""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT model_name, usage_count, last_used, 
                           total_duration, average_processing_time
                    FROM model_usage
                    ORDER BY usage_count DESC
                ''')
                
                rows = cursor.fetchall()
                conn.close()
                
                stats = {}
                for row in rows:
                    stats[row[0]] = {
                        "usage_count": row[1],
                        "last_used": row[2],
                        "total_duration": row[3],
                        "average_processing_time": row[4]
                    }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting model statistics: {str(e)}")
            return {}
    
    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """Search synthesis history"""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                search_pattern = f"%{query}%"
                cursor.execute('''
                    SELECT id, text, audio_path, settings, created_at,
                           model_name, speaker, language
                    FROM synthesis_history
                    WHERE text LIKE ? OR model_name LIKE ? OR speaker LIKE ?
                    ORDER BY created_at DESC
                    LIMIT 100
                ''', (search_pattern, search_pattern, search_pattern))
                
                rows = cursor.fetchall()
                conn.close()
                
                history = []
                for row in rows:
                    history.append({
                        "id": row[0],
                        "text": row[1],
                        "audio_path": row[2],
                        "settings": json.loads(row[3]),
                        "created_at": row[4],
                        "model_name": row[5],
                        "speaker": row[6],
                        "language": row[7]
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Error searching history: {str(e)}")
            return []
    
    def cleanup_old_records(self, days: int = 30):
        """Clean up old records"""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Delete old synthesis history
                cursor.execute('''
                    DELETE FROM synthesis_history
                    WHERE created_at < datetime('now', '-{} days')
                '''.format(days))
                
                deleted_count = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                logger.info(f"Cleaned up {deleted_count} old records")
                
        except Exception as e:
            logger.error(f"Error cleaning up records: {str(e)}")
    
    def export_data(self, export_path: str) -> bool:
        """Export database data to JSON"""
        try:
            export_data = {
                "synthesis_history": self.get_history(limit=10000),
                "model_statistics": self.get_model_statistics(),
                "export_date": datetime.now().isoformat()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Data exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return False
    
    def import_data(self, import_path: str) -> bool:
        """Import data from JSON file"""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            # Import synthesis history
            if "synthesis_history" in import_data:
                with self.lock:
                    conn = sqlite3.connect(str(self.db_path))
                    cursor = conn.cursor()
                    
                    for record in import_data["synthesis_history"]:
                        cursor.execute('''
                            INSERT OR IGNORE INTO synthesis_history 
                            (id, text, audio_path, settings, model_name, speaker, language)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            record["id"],
                            record["text"],
                            record["audio_path"],
                            json.dumps(record["settings"]),
                            record["model_name"],
                            record["speaker"],
                            record["language"]
                        ))
                    
                    conn.commit()
                    conn.close()
            
            logger.info(f"Data imported from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing data: {str(e)}")
            return False