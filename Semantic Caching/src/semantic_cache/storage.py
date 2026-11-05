"""Transactional signed SQLite entries; keys are supplied outside cache storage."""
import hashlib
import hmac
import json
import sqlite3
from threading import RLock

class SQLiteStore:
    def __init__(self, path, key):
        if len(key)<32:
            raise ValueError('A signing key of at least 32 bytes is required.')
        self.key = key
        self.lock = RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, payload TEXT NOT NULL, signature TEXT NOT NULL)')

    def put(self, entry):
        payload = json.dumps(entry,sort_keys=True,separators=(',',':'),allow_nan=False)
        signature = hmac.new(self.key,payload.encode(),hashlib.sha256).hexdigest()
        with self.lock, self.db:
            self.db.execute('INSERT OR REPLACE INTO entries VALUES (?,?,?)',(entry['key'],payload,signature))

    def delete(self, key):
        with self.lock, self.db:
            self.db.execute('DELETE FROM entries WHERE id=?',(key,))

    def load(self):
        with self.lock:
            rows = list(self.db.execute('SELECT id,payload,signature FROM entries'))
        entries = []
        for key,payload,signature in rows:
            expected = hmac.new(self.key,payload.encode(),hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected,signature):
                raise ValueError('Cache integrity verification failed.')
            entry = json.loads(payload)
            if entry['key']!=key:
                raise ValueError('Cache key does not match its signed payload.')
            entries.append(entry)
        return entries

    def close(self):
        with self.lock:
            self.db.close()
