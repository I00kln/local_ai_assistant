# sqlite_store.py
# L3 长期记忆存储层 - SQLite 实现
import os
import sqlite3
import json
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from config import config
from memory_tags import MemoryTags


class EncryptionError(Exception):
    """加密错误"""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def get_encryption_key() -> Optional[str]:
    """
    从环境变量获取加密密钥
    
    Returns:
        加密密钥或 None
    """
    key_env = getattr(config.sqlite, 'encryption_key_env', 'SQLITE_ENCRYPTION_KEY')
    return os.environ.get(key_env)


def is_sqlcipher_available() -> bool:
    """
    检查 SQLCipher 是否可用
    
    Returns:
        SQLCipher 是否可用
    """
    try:
        import sqlcipher3
        return True
    except ImportError:
        try:
            from pysqlcipher3 import dbapi2
            return True
        except ImportError:
            return False


@dataclass
class MemoryRecord:
    """记忆记录结构"""
    id: Optional[int] = None
    text: str = ""
    compressed_text: Optional[str] = None
    source: str = "user"
    weight: float = 1.0
    access_count: int = 0
    last_access_time: Optional[str] = None
    created_time: Optional[str] = None
    metadata: Dict[str, Any] = None
    vector_id: Optional[str] = None
    is_vectorized: int = 0
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now().isoformat()
        if self.last_access_time is None:
            self.last_access_time = self.created_time
        if self.metadata is None:
            self.metadata = {}


class SQLiteStore:
    """
    L3 长期记忆存储层
    
    功能：
    - 持久化存储压缩后的记忆
    - 权重管理（访问增加权重，时间降低权重）
    - 定期遗忘机制
    - 全文搜索支持
    
    并发策略：
    - 使用 threading.local() 为每个线程维护独立连接
    - WAL 模式下读操作可并发执行
    - 仅写操作使用互斥锁保护
    """
    
    WEIGHT_DECAY_RATE = 0.95
    WEIGHT_BOOST_ON_ACCESS = 1.2
    MIN_WEIGHT_THRESHOLD = 0.3
    MAX_WEIGHT = 5.0
    
    FORGET_CHECK_INTERVAL = 3600
    FORGET_AGE_DAYS = 30
    
    MAX_POOL_SIZE = 10
    CONNECTION_TIMEOUT = 3600
    BACKUP_SUFFIX = ".backup"
    CONNECTION_HEALTH_CHECK_INTERVAL = 3600
    CONNECTION_IDLE_TIMEOUT = 3600
    BACKUP_INTERVAL = 3600
    MAX_BACKUP_VERSIONS = 3
    MAX_RETRY_ATTEMPTS = 2
    
    def __init__(self, db_path: str = "memory.db", encryption_key: str = None):
        self.db_path = db_path
        self._write_lock = threading.Lock()
        self.lock = self._write_lock
        self._local = threading.local()
        self._last_cleanup = datetime.now()
        self._integrity_checked = False
        self._last_connection_check: Dict[int, float] = {}
        self._last_backup_time: float = 0
        
        self._encryption_enabled = False
        self._encryption_key = None
        
        encryption_configured = getattr(config.sqlite, 'encryption_enabled', False)
        if encryption_configured:
            key = encryption_key or get_encryption_key()
            if key and is_sqlcipher_available():
                self._encryption_enabled = True
                self._encryption_key = key
                print("[安全] SQLite 加密已启用 (SQLCipher)")
            elif encryption_configured and not key:
                print("[警告] SQLite 加密已配置但未找到密钥，请设置 SQLITE_ENCRYPTION_KEY 环境变量")
            elif encryption_configured and not is_sqlcipher_available():
                print("[警告] SQLite 加密已配置但 SQLCipher 不可用，请安装: pip install sqlcipher3")
        
        self._init_database()
    
    def _validate_connection(self, conn) -> bool:
        """
        验证连接是否有效
        
        Args:
            conn: SQLite 连接
        
        Returns:
            连接是否有效
        """
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return True
        except (sqlite3.OperationalError, sqlite3.DatabaseError, 
                sqlite3.InterfaceError, sqlite3.ProgrammingError) as e:
            self._log_connection_error(e)
            return False
    
    def _log_connection_error(self, error: Exception):
        """记录连接错误"""
        import sys
        error_type = type(error).__name__
        print(f"[SQLite] 连接验证失败 ({error_type}): {error}", file=sys.stderr)
    
    def _create_connection(self):
        """
        创建数据库连接（支持加密）
        
        Returns:
            SQLite 连接对象
        """
        if self._encryption_enabled:
            try:
                import sqlcipher3
                conn = sqlcipher3.connect(self.db_path)
            except ImportError:
                try:
                    from pysqlcipher3 import dbapi2
                    conn = dbapi2.connect(self.db_path)
                except ImportError:
                    raise EncryptionError(
                        "SQLCIPHER_UNAVAILABLE",
                        "SQLCipher 不可用，请安装 sqlcipher3 或 pysqlcipher3"
                    )
            
            conn.execute(f"PRAGMA key='{self._encryption_key}'")
            conn.execute("PRAGMA cipher_compatibility=4")
            
            try:
                conn.execute("SELECT count(*) FROM sqlite_master")
            except Exception as e:
                raise EncryptionError(
                    "DECRYPTION_FAILED",
                    f"数据库解密失败，请检查密钥是否正确: {e}"
                )
        else:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
        
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA busy_timeout=30000")
        
        return conn
    
    def _get_read_connection(self):
        """
        获取读连接（线程本地，无需锁）
        
        WAL 模式下，多个读连接可以并发执行
        
        增强：
        - 连接有效性检测（SELECT 1）
        - 自动重连机制
        - 定期健康检查
        - 空闲超时自动关闭并重建
        - 捕获 ProgrammingError 并自动恢复
        
        Returns:
            有效的 SQLite 连接
        """
        current_thread_id = threading.get_ident()
        current_time = time.time()
        
        need_new_connection = False
        
        if not hasattr(self._local, 'read_conn') or self._local.read_conn is None:
            need_new_connection = True
        else:
            last_use_time = getattr(self._local, 'last_use_time', current_time)
            if current_time - last_use_time > self.CONNECTION_IDLE_TIMEOUT:
                try:
                    self._local.read_conn.close()
                except Exception:
                    pass
                self._local.read_conn = None
                need_new_connection = True
                print(f"[SQLite] 线程 {current_thread_id} 连接空闲超时，正在重建...")
            else:
                last_check = self._last_connection_check.get(current_thread_id, 0)
                if current_time - last_check > self.CONNECTION_HEALTH_CHECK_INTERVAL:
                    if not self._validate_connection(self._local.read_conn):
                        try:
                            self._local.read_conn.close()
                        except Exception:
                            pass
                        self._local.read_conn = None
                        need_new_connection = True
                        print(f"[SQLite] 线程 {current_thread_id} 连接失效，正在重连...")
                    else:
                        self._last_connection_check[current_thread_id] = current_time
        
        if need_new_connection:
            self._local.read_conn = self._create_connection()
            self._last_connection_check[current_thread_id] = current_time
        
        self._local.last_use_time = current_time
        
        return self._local.read_conn
    
    @contextmanager
    def _get_write_connection(self):
        """
        获取写连接（需要锁保护）
        
        写操作需要互斥锁保护，确保数据一致性
        """
        with self._write_lock:
            conn = self._get_read_connection()
            try:
                yield conn
            except sqlite3.Error as e:
                try:
                    conn.rollback()
                except:
                    pass
                raise
    
    def _retry_on_connection_error(self, func, *args, **kwargs):
        """
        连接错误自动重试装饰器
        
        捕获 ProgrammingError 和其他连接相关错误，
        自动重建连接并重试操作。
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            函数执行结果
        
        Raises:
            最后一次重试失败后的异常
        """
        last_error = None
        
        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except sqlite3.ProgrammingError as e:
                last_error = e
                error_msg = str(e).lower()
                
                if "closed" in error_msg or "cannot operate" in error_msg:
                    print(f"[SQLite] 检测到连接已关闭 (尝试 {attempt + 1}/{self.MAX_RETRY_ATTEMPTS})，正在重建连接...")
                    
                    if hasattr(self._local, 'read_conn') and self._local.read_conn:
                        try:
                            self._local.read_conn.close()
                        except Exception:
                            pass
                        self._local.read_conn = None
                    
                    self._local.read_conn = self._create_connection()
                    current_thread_id = threading.get_ident()
                    self._last_connection_check[current_thread_id] = time.time()
                else:
                    raise
            except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
                last_error = e
                print(f"[SQLite] 连接错误 (尝试 {attempt + 1}/{self.MAX_RETRY_ATTEMPTS}): {e}")
                
                if hasattr(self._local, 'read_conn') and self._local.read_conn:
                    try:
                        self._local.read_conn.close()
                    except Exception:
                        pass
                    self._local.read_conn = None
        
        if last_error:
            raise last_error
    
    def _check_integrity(self) -> bool:
        """
        检查数据库完整性
        
        Returns:
            True: 数据库完整
            False: 数据库损坏
        """
        try:
            conn = self._create_connection()
            cursor = conn.cursor()
            result = cursor.execute("PRAGMA integrity_check").fetchone()
            conn.close()
            
            if result and result[0] == "ok":
                return True
            else:
                print(f"[警告] 数据库完整性检查失败: {result}")
                return False
        except EncryptionError as e:
            print(f"[错误] 加密数据库访问失败: {e}")
            return False
        except sqlite3.DatabaseError as e:
            print(f"[错误] 数据库损坏: {e}")
            return False
        except Exception as e:
            print(f"[错误] 完整性检查异常: {e}")
            return True
    
    def _try_recover_from_backup(self) -> bool:
        """
        尝试从备份恢复数据库
        
        Returns:
            True: 恢复成功
            False: 无备份或恢复失败
        """
        import shutil
        import os
        
        backup_path = self.db_path + self.BACKUP_SUFFIX
        
        if not os.path.exists(backup_path):
            print(f"[警告] 无备份文件: {backup_path}")
            return False
        
        try:
            if os.path.exists(self.db_path):
                corrupt_backup = self.db_path + ".corrupt"
                shutil.move(self.db_path, corrupt_backup)
                print(f"[备份] 损坏数据库已保存为: {corrupt_backup}")
            
            shutil.copy(backup_path, self.db_path)
            print(f"[恢复] 从备份恢复成功: {backup_path}")
            return True
            
        except Exception as e:
            print(f"[错误] 从备份恢复失败: {e}")
            return False
    
    def _create_backup(self, force: bool = False):
        """
        创建数据库备份（支持多版本）
        
        Args:
            force: 是否强制创建（忽略时间间隔）
        
        备份策略：
        - 每小时自动备份一次
        - 保留最近3个版本的备份
        - 备份文件命名：memory.db.backup.1, memory.db.backup.2, memory.db.backup.3
        """
        import shutil
        import os
        
        current_time = time.time()
        
        if not force and current_time - self._last_backup_time < self.BACKUP_INTERVAL:
            return
        
        try:
            if not os.path.exists(self.db_path):
                return
            
            backup_dir = os.path.dirname(self.db_path) or "."
            base_name = os.path.basename(self.db_path)
            
            oldest_backup = os.path.join(backup_dir, f"{base_name}.backup.{self.MAX_BACKUP_VERSIONS}")
            if os.path.exists(oldest_backup):
                os.remove(oldest_backup)
            
            for i in range(self.MAX_BACKUP_VERSIONS - 1, 0, -1):
                old_backup = os.path.join(backup_dir, f"{base_name}.backup.{i}")
                new_backup = os.path.join(backup_dir, f"{base_name}.backup.{i + 1}")
                if os.path.exists(old_backup):
                    shutil.move(old_backup, new_backup)
            
            latest_backup = os.path.join(backup_dir, f"{base_name}.backup.1")
            shutil.copy2(self.db_path, latest_backup)
            
            main_backup = self.db_path + self.BACKUP_SUFFIX
            shutil.copy2(self.db_path, main_backup)
            
            self._last_backup_time = current_time
            print(f"[备份] 数据库备份完成: {latest_backup}")
            
        except Exception as e:
            print(f"[警告] 创建备份失败: {e}")
    
    def _try_recover_from_backup_versions(self) -> bool:
        """
        尝试从多版本备份恢复
        
        按版本顺序尝试恢复，直到找到有效的备份
        
        Returns:
            是否恢复成功
        """
        import shutil
        import os
        
        backup_dir = os.path.dirname(self.db_path) or "."
        base_name = os.path.basename(self.db_path)
        
        for version in range(1, self.MAX_BACKUP_VERSIONS + 1):
            backup_path = os.path.join(backup_dir, f"{base_name}.backup.{version}")
            
            if not os.path.exists(backup_path):
                continue
            
            try:
                if os.path.exists(self.db_path):
                    corrupt_backup = self.db_path + f".corrupt.{int(time.time())}"
                    shutil.move(self.db_path, corrupt_backup)
                    print(f"[备份] 损坏数据库已保存为: {corrupt_backup}")
                
                shutil.copy2(backup_path, self.db_path)
                
                if self._check_integrity():
                    print(f"[恢复] 从备份版本 {version} 恢复成功")
                    return True
                else:
                    print(f"[警告] 备份版本 {version} 也已损坏，尝试下一版本")
                    
            except Exception as e:
                print(f"[错误] 从备份版本 {version} 恢复失败: {e}")
        
        return False
    
    def _init_database(self):
        """
        初始化数据库表结构
        
        包含完整性检查和自动恢复：
        1. 检查数据库文件是否存在
        2. 检查数据库完整性
        3. 损坏时尝试从多版本备份恢复
        4. 创建表结构
        5. 创建初始备份
        """
        import os
        
        if os.path.exists(self.db_path) and not self._integrity_checked:
            self._integrity_checked = True
            
            if not self._check_integrity():
                print("[警告] 数据库损坏，尝试从多版本备份恢复...")
                
                if self._try_recover_from_backup_versions():
                    print("[成功] 从备份恢复成功")
                elif self._try_recover_from_backup():
                    print("[成功] 从主备份恢复成功")
                else:
                    print("[严重] 无可用备份，将重建数据库")
                    self._rebuild_database()
        
        try:
            with self._get_write_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        compressed_text TEXT,
                        source TEXT DEFAULT 'user',
                        weight REAL DEFAULT 1.0,
                        access_count INTEGER DEFAULT 0,
                        last_access_time TEXT,
                        created_time TEXT,
                        metadata TEXT,
                        is_archived INTEGER DEFAULT 0,
                        vector_id TEXT,
                        is_vectorized INTEGER DEFAULT 0
                    )
                """)
                
                try:
                    cursor.execute("ALTER TABLE memories ADD COLUMN vector_id TEXT")
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute("ALTER TABLE memories ADD COLUMN is_vectorized INTEGER DEFAULT 0")
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute("ALTER TABLE memories ADD COLUMN text_hash TEXT")
                except sqlite3.OperationalError:
                    pass
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pending_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data TEXT NOT NULL,
                        created_time TEXT NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'pending'
                    )
                """)
                
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pending_queue_status ON pending_queue(status, created_time)")
                except sqlite3.OperationalError:
                    pass
                
                conn.commit()
                
                if os.path.exists(self.db_path):
                    self._create_backup()
                    
        except sqlite3.DatabaseError as e:
            print(f"[严重] 数据库初始化失败: {e}")
            self._rebuild_database()
    
    def _rebuild_database(self):
        """
        重建数据库
        
        当数据库损坏且无法恢复时，删除并重建
        """
        import os
        import shutil
        
        try:
            if os.path.exists(self.db_path):
                corrupt_path = self.db_path + ".corrupt." + str(int(time.time()))
                shutil.move(self.db_path, corrupt_path)
                print(f"[备份] 损坏数据库已保存为: {corrupt_path}")
            
            with self._get_write_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        compressed_text TEXT,
                        source TEXT DEFAULT 'user',
                        weight REAL DEFAULT 1.0,
                        access_count INTEGER DEFAULT 0,
                        last_access_time TEXT,
                        created_time TEXT,
                        metadata TEXT,
                        is_archived INTEGER DEFAULT 0,
                        vector_id TEXT,
                        is_vectorized INTEGER DEFAULT 0,
                        text_hash TEXT
                    )
                """)
                conn.commit()
            
            print("[完成] 数据库重建成功")
            
        except Exception as e:
            print(f"[严重] 数据库重建失败: {e}")
            raise
            
            # 全文搜索虚拟表（FTS5）
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    text,
                    compressed_text,
                    content='memories',
                    content_rowid='id'
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_weight 
                ON memories(weight DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_time 
                ON memories(created_time DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_access 
                ON memories(last_access_time DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vector_id 
                ON memories(vector_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_is_vectorized 
                ON memories(is_vectorized)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_hash 
                ON memories(text_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decay_candidate 
                ON memories(is_archived, created_time, weight)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_forget_candidate 
                ON memories(is_archived, weight, is_vectorized)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_weight_access 
                ON memories(weight DESC, last_access_time DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vectorized_weight 
                ON memories(is_vectorized, weight DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_weight 
                ON memories(source, weight DESC)
            """)
            
            # 触发器：自动同步FTS索引
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, text, compressed_text)
                    VALUES (new.id, new.text, COALESCE(new.compressed_text, ''));
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, text, compressed_text)
                    VALUES('delete', old.id, old.text, COALESCE(old.compressed_text, ''));
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, text, compressed_text)
                    VALUES('delete', old.id, old.text, COALESCE(old.compressed_text, ''));
                    INSERT INTO memories_fts(rowid, text, compressed_text)
                    VALUES (new.id, new.text, COALESCE(new.compressed_text, ''));
                END
            """)
            
            # 会话历史表（用于UI分页显示）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    assistant_response TEXT,
                    source TEXT DEFAULT 'local',
                    timestamp TEXT,
                    metadata TEXT
                )
            """)
            
            # 会话历史索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id 
                ON conversations(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_time 
                ON conversations(timestamp DESC)
            """)
            
            conn.commit()
            print(f"SQLite数据库初始化完成: {self.db_path}")
    
    def add(self, record: MemoryRecord) -> int:
        """添加记忆记录"""
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories 
                (text, compressed_text, source, weight, access_count, 
                 last_access_time, created_time, metadata, vector_id, is_vectorized)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.text,
                record.compressed_text,
                record.source,
                record.weight,
                record.access_count,
                record.last_access_time,
                record.created_time,
                json.dumps(record.metadata, ensure_ascii=False),
                record.vector_id,
                record.is_vectorized
            ))
            conn.commit()
            
            self._create_backup()
            
            return cursor.lastrowid
    
    def add_batch(self, records: List[MemoryRecord]) -> List[int]:
        """批量添加记忆（优化版：使用 executemany）"""
        if not records:
            return []
        
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT MAX(id) FROM memories")
            max_id_before = cursor.fetchone()[0] or 0
            
            data = [(
                record.text,
                record.compressed_text,
                record.source,
                record.weight,
                record.access_count,
                record.last_access_time,
                record.created_time,
                json.dumps(record.metadata, ensure_ascii=False),
                record.vector_id,
                record.is_vectorized
            ) for record in records]
            
            cursor.executemany("""
                INSERT INTO memories 
                (text, compressed_text, source, weight, access_count, 
                 last_access_time, created_time, metadata, vector_id, is_vectorized)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            
            conn.commit()
            
            return list(range(max_id_before + 1, max_id_before + 1 + len(records)))
    
    def get(self, record_id: int) -> Optional[MemoryRecord]:
        """
        根据ID获取记录
        
        增强功能：
        - 自动重试连接错误
        - 捕获 ProgrammingError 并恢复
        """
        def _do_get():
            conn = self._get_read_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
            return None
        
        return self._retry_on_connection_error(_do_get)
    
    def search(self, query: str, limit: int = 10, min_weight: float = 0.0) -> List[MemoryRecord]:
        """
        全文搜索
        
        使用FTS5进行全文搜索，返回匹配的记忆
        
        增强功能：
        - 自动重试连接错误
        - 捕获 ProgrammingError 并恢复
        """
        def _do_search():
            conn = self._get_read_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT m.* FROM memories m
                JOIN memories_fts fts ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
                AND m.weight >= ?
                AND m.is_archived = 0
                ORDER BY m.weight DESC, bm25(memories_fts) ASC
                LIMIT ?
            """, (query, min_weight, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append(self._row_to_record(row))
            
            return results
        
        return self._retry_on_connection_error(_do_search)
    
    def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[MemoryRecord]:
        """
        关键词搜索（更宽松的匹配）
        
        使用LIKE进行模糊匹配
        
        增强功能：
        - 自动重试连接错误
        - 捕获 ProgrammingError 并恢复
        """
        def _do_search_by_keywords():
            conn = self._get_read_connection()
            cursor = conn.cursor()
            
            conditions = []
            params = []
            for kw in keywords:
                conditions.append("(text LIKE ? OR compressed_text LIKE ?)")
                params.extend([f"%{kw}%", f"%{kw}%"])
            
            query = f"""
                SELECT * FROM memories 
                WHERE ({' OR '.join(conditions)})
                AND is_archived = 0
                ORDER BY weight DESC, last_access_time DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append(self._row_to_record(row))
            
            return results
        
        return self._retry_on_connection_error(_do_search_by_keywords)
    
    def update_weight(self, record_id: int, delta: float = None, boost: bool = False):
        """
        更新记录权重
        
        Args:
            record_id: 记录ID
            delta: 权重变化量（正数增加，负数减少）
            boost: 是否为访问提升（使用固定提升比例）
        """
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            
            if boost:
                cursor.execute("""
                    UPDATE memories 
                    SET weight = MIN(?, weight * ?),
                        access_count = access_count + 1,
                        last_access_time = ?
                    WHERE id = ?
                """, (self.MAX_WEIGHT, self.WEIGHT_BOOST_ON_ACCESS, 
                      datetime.now().isoformat(), record_id))
            else:
                cursor.execute("""
                    UPDATE memories 
                    SET weight = MAX(0, MIN(?, weight + ?))
                    WHERE id = ?
                """, (self.MAX_WEIGHT, delta or 0, record_id))
            
            conn.commit()
    
    def update_metadata_field(
        self, 
        record_id: int, 
        field_path: str, 
        field_value: Any
    ) -> bool:
        """
        原子更新 metadata 中的特定字段
        
        使用 SQLite json_set 函数进行原子操作，避免并发覆盖
        
        Args:
            record_id: 记录ID
            field_path: 字段路径（如 "semantic_tag" 或 "tags.custom"）
            field_value: 字段值
        
        Returns:
            是否成功
        """
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            
            try:
                json_path = f"$.{field_path}"
                value_json = json.dumps(field_value, ensure_ascii=False)
                
                cursor.execute("""
                    UPDATE memories 
                    SET metadata = json_set(
                        COALESCE(metadata, '{}'),
                        ?, ?
                    )
                    WHERE id = ?
                """, (json_path, value_json, record_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
            except sqlite3.OperationalError as e:
                if "no such function: json_set" in str(e):
                    return self._update_metadata_field_fallback(
                        record_id, field_path, field_value
                    )
                raise
            except Exception as e:
                print(f"原子更新 metadata 字段失败: {e}")
                return False
    
    def _update_metadata_field_fallback(
        self, 
        record_id: int, 
        field_path: str, 
        field_value: Any
    ) -> bool:
        """
        原子更新 metadata 字段的回退方案
        
        当 SQLite 不支持 json_set 时使用
        使用锁保护读取-修改-写入操作
        """
        with self.lock:
            record = self.get(record_id)
            if not record:
                return False
            
            if record.metadata is None:
                record.metadata = {}
            
            path_parts = field_path.split(".")
            current = record.metadata
            
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[path_parts[-1]] = field_value
            
            return self.update_metadata(record_id, record.metadata)
    
    def update_metadata(self, record_id: int, metadata: Dict) -> bool:
        """
        更新整个 metadata
        
        Args:
            record_id: 记录ID
            metadata: 新的 metadata
        
        Returns:
            是否成功
        """
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE memories 
                SET metadata = ?
                WHERE id = ?
            """, (json.dumps(metadata, ensure_ascii=False), record_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def decay_weights(self, days_threshold: int = None, batch_size: int = 1000) -> Dict[str, Any]:
        """
        权重衰减（遗忘机制）
        
        对超过指定天数的记忆进行权重衰减
        注意：带有"important"标签的记忆不会被衰减或删除
        
        优化：
        - 分批处理大量记录，避免长时间锁表
        - 使用复合索引加速查询
        - 排除重要记忆（metadata.tags 包含 "important"）
        
        Args:
            days_threshold: 衰减阈值天数
            batch_size: 每批处理的最大记录数
        
        Returns:
            {
                "decayed": 衰减数量,
                "forgotten": 遗忘数量,
                "vector_ids_to_delete": 需要删除的ChromaDB向量ID列表
            }
        """
        if days_threshold is None:
            days_threshold = self.FORGET_AGE_DAYS
        
        total_decayed = 0
        total_forgotten = 0
        all_vector_ids_to_delete = []
        
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            
            threshold_time = (datetime.now() - timedelta(days=days_threshold)).isoformat()
            
            while True:
                cursor.execute("""
                    UPDATE memories 
                    SET weight = weight * ?
                    WHERE id IN (
                        SELECT id FROM memories 
                        WHERE created_time < ?
                        AND is_archived = 0
                        AND weight > ?
                        AND (metadata IS NULL OR metadata NOT LIKE '%"important"%')
                        LIMIT ?
                    )
                """, (self.WEIGHT_DECAY_RATE, threshold_time, 
                      self.MIN_WEIGHT_THRESHOLD, batch_size))
                
                batch_decayed = cursor.rowcount
                total_decayed += batch_decayed
                
                if batch_decayed < batch_size:
                    break
            
            while True:
                cursor.execute("""
                    SELECT id, vector_id FROM memories 
                    WHERE weight < ?
                    AND is_archived = 0
                    AND is_vectorized = 1
                    AND vector_id IS NOT NULL
                    AND vector_id != ''
                    AND (metadata IS NULL OR metadata NOT LIKE '%"important"%')
                    LIMIT ?
                """, (self.MIN_WEIGHT_THRESHOLD, batch_size))
                
                rows = cursor.fetchall()
                if not rows:
                    break
                
                ids_to_delete = [row[0] for row in rows]
                vector_ids = [row[1] for row in rows if row[1]]
                all_vector_ids_to_delete.extend(vector_ids)
                
                placeholders = ",".join("?" * len(ids_to_delete))
                cursor.execute(f"""
                    DELETE FROM memories 
                    WHERE id IN ({placeholders})
                """, ids_to_delete)
                
                total_forgotten += len(ids_to_delete)
            
            conn.commit()
            
            if total_decayed > 0 or total_forgotten > 0:
                print(f"权重衰减完成: {total_decayed} 条记忆衰减, {total_forgotten} 条记忆遗忘")
            
            return {
                "decayed": total_decayed,
                "forgotten": total_forgotten,
                "vector_ids_to_delete": all_vector_ids_to_delete
            }
    
    def get_low_weight_memories(self, threshold: float = None, limit: int = 100) -> List[MemoryRecord]:
        """
        获取低权重记忆（准备遗忘或压缩）
        
        条件：
        - 权重低于阈值
        - 已向量化（is_vectorized=1，L2中的记录）
        - 未归档
        - 不在迁移中（is_vectorized != 3）
        
        增强功能：
        - 自动重试连接错误
        - 捕获 ProgrammingError 并恢复
        """
        if threshold is None:
            threshold = self.MIN_WEIGHT_THRESHOLD * 2
        
        def _do_get():
            conn = self._get_read_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM memories 
                WHERE weight < ?
                AND is_archived = 0
                AND is_vectorized = 1
                ORDER BY weight ASC, last_access_time ASC
                LIMIT ?
            """, (threshold, limit))
            
            return [self._row_to_record(row) for row in cursor.fetchall()]
        
        return self._retry_on_connection_error(_do_get)
    
    def get_high_weight_memories(self, threshold: float = None, limit: int = 50) -> List[MemoryRecord]:
        """
        获取高权重记忆（准备回流到L2）
        
        条件：
        - 权重超过阈值
        - 最近有访问
        - 未向量化（is_vectorized=0，L3中的记录）
        - 未归档
        - 不在迁移中（is_vectorized != 3）
        
        增强功能：
        - 自动重试连接错误
        - 捕获 ProgrammingError 并恢复
        """
        if threshold is None:
            threshold = self.MAX_WEIGHT * 0.6
        
        recent_time = (datetime.now() - timedelta(days=7)).isoformat()
        
        def _do_get():
            conn = self._get_read_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM memories 
                WHERE weight >= ?
                AND last_access_time > ?
                AND is_vectorized = 0
                AND is_archived = 0
                ORDER BY weight DESC, last_access_time DESC
                LIMIT ?
            """, (threshold, recent_time, limit))
            
            return [self._row_to_record(row) for row in cursor.fetchall()]
        
        return self._retry_on_connection_error(_do_get)
    
    def get_recent_memories(
        self, 
        limit: int = 100, 
        vectorized_only: bool = False
    ) -> List[MemoryRecord]:
        """
        获取最近的记忆（用于合并检测）
        
        Args:
            limit: 返回数量限制
            vectorized_only: 是否只返回已向量化的记忆（L2）
        
        Returns:
            最近的记忆记录列表
        """
        conn = self._get_read_connection()
        cursor = conn.cursor()
        
        if vectorized_only:
            cursor.execute("""
                SELECT * FROM memories 
                WHERE is_vectorized = 1
                AND is_archived = 0
                ORDER BY created_time DESC
                LIMIT ?
            """, (limit,))
        else:
            cursor.execute("""
                SELECT * FROM memories 
                WHERE is_archived = 0
                ORDER BY created_time DESC
                LIMIT ?
            """, (limit,))
        
        return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def get_upgradeable_sqlite_only(self, threshold: float = None, limit: int = 20) -> List[MemoryRecord]:
        """
        获取可升级的 sqlite_only 记录
        
        条件：
        - 权重超过阈值（说明有价值）
        - 最近有访问（说明仍在使用）
        - is_vectorized = -1（sqlite_only 标记）
        
        这些记录最初被判定为低价值，但后续访问表明有价值，
        应该升级到向量库以支持语义检索。
        """
        if threshold is None:
            threshold = self.MAX_WEIGHT * 0.5
        
        recent_time = (datetime.now() - timedelta(days=7)).isoformat()
        
        conn = self._get_read_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM memories 
            WHERE weight >= ?
            AND last_access_time > ?
            AND is_vectorized = -1
            AND is_archived = 0
            ORDER BY weight DESC, last_access_time DESC
            LIMIT ?
        """, (threshold, recent_time, limit))
        
        return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def get_unaccessed_memories(self, days: int = 7, limit: int = 100) -> List[MemoryRecord]:
        """获取长时间未访问的记忆"""
        threshold_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = self._get_read_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM memories 
            WHERE last_access_time < ?
            AND is_archived = 0
            ORDER BY last_access_time ASC
            LIMIT ?
        """, (threshold_time, limit))
        
        return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def mark_important(self, record_id: int, important: bool = True):
        """
        标记记忆为重要/取消重要
        
        重要记忆不会被权重衰减或遗忘
        """
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT metadata FROM memories WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            
            if row:
                metadata = json.loads(row[0]) if row[0] else {}
                tags = metadata.get("tags", [])
                
                if important:
                    if "important" not in tags:
                        tags.append("important")
                else:
                    if "important" in tags:
                        tags.remove("important")
                
                metadata[MemoryTags.TAGS] = tags
                
                cursor.execute("""
                    UPDATE memories SET metadata = ? WHERE id = ?
                """, (json.dumps(metadata, ensure_ascii=False), record_id))
                conn.commit()
                
                return True
            return False
    
    def archive_memory(self, record_id: int):
        """归档记忆（软删除）"""
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE memories SET is_archived = 1 WHERE id = ?
            """, (record_id,))
            conn.commit()
    
    def delete_memory(self, record_id: int):
        """永久删除记忆"""
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (record_id,))
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        conn = self._get_read_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memories WHERE is_archived = 0")
        total_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(weight) FROM memories WHERE is_archived = 0")
        avg_weight = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT COUNT(*) FROM memories 
            WHERE weight > 2.0 AND is_archived = 0
        """)
        high_weight_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM memories 
            WHERE weight < ? AND is_archived = 0
        """, (self.MIN_WEIGHT_THRESHOLD * 2,))
        low_weight_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM memories 
            WHERE last_access_time > ?
            AND is_archived = 0
        """, ((datetime.now() - timedelta(days=1)).isoformat(),))
        recent_access_count = cursor.fetchone()[0]
        
        return {
            "total_count": total_count,
            "avg_weight": round(avg_weight, 3),
            "high_weight_count": high_weight_count,
            "low_weight_count": low_weight_count,
            "recent_access_count": recent_access_count,
            "db_size_mb": round(os.path.getsize(self.db_path) / 1024 / 1024, 2) 
                          if os.path.exists(self.db_path) else 0
        }
    
    def vacuum(self):
        """清理数据库碎片"""
        with self._get_write_connection() as conn:
            conn.execute("VACUUM")
            print("数据库清理完成")
    
    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        """将数据库行转换为MemoryRecord"""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass
        
        return MemoryRecord(
            id=row["id"],
            text=row["text"],
            compressed_text=row["compressed_text"],
            source=row["source"],
            weight=row["weight"],
            access_count=row["access_count"],
            last_access_time=row["last_access_time"],
            created_time=row["created_time"],
            metadata=metadata,
            vector_id=row["vector_id"] if "vector_id" in row.keys() else None,
            is_vectorized=row["is_vectorized"] if "is_vectorized" in row.keys() else 0
        )
    
    def update_vector_status(self, record_id: int, vector_id: str, is_vectorized: int = 1):
        """
        更新记录的向量化状态
        
        Args:
            record_id: SQLite记录ID
            vector_id: ChromaDB中的文档ID
            is_vectorized: 向量化状态 (1=成功, -1=失败, 0=未处理)
        """
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE memories 
                SET vector_id = ?, is_vectorized = ?
                WHERE id = ?
            """, (vector_id, is_vectorized, record_id))
            conn.commit()
    
    def get_unvectorized(self, limit: int = 50) -> List[MemoryRecord]:
        """
        获取未向量化的记录（用于重试）
        
        包括：
        - is_vectorized=0: 未处理
        - is_vectorized=2: 迁移中断，需要恢复
        
        增强功能：
        - 自动重试连接错误
        - 捕获 ProgrammingError 并恢复
        """
        def _do_get():
            conn = self._get_read_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM memories 
                WHERE is_vectorized IN (0, 2)
                AND is_archived = 0
                ORDER BY created_time DESC
                LIMIT ?
            """, (limit,))
            return [self._row_to_record(row) for row in cursor.fetchall()]
        
        return self._retry_on_connection_error(_do_get)
    
    def get_records_by_vector_status(self, is_vectorized: int, limit: int = 50) -> List[MemoryRecord]:
        """
        根据向量化状态获取记录
        
        Args:
            is_vectorized: 向量化状态
                - 0: 未向量化(L3)
                - 1: 已向量化(L2)
                - -1: 向量化失败（待重试）
                - 2: 向量化失败
                - 3: 迁移中
            limit: 返回数量限制
        
        Returns:
            符合条件的记录列表
        """
        conn = self._get_read_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM memories 
            WHERE is_vectorized = ?
            AND is_archived = 0
            ORDER BY created_time DESC
            LIMIT ?
        """, (is_vectorized, limit))
        return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def get_failed_vectorizations(self, max_retries: int = 3, limit: int = 20) -> List[MemoryRecord]:
        """
        获取向量化失败的记录（带重试次数限制）
        
        Args:
            max_retries: 最大重试次数
            limit: 返回数量限制
        
        Returns:
            可重试的失败记录列表
        """
        conn = self._get_read_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM memories 
            WHERE is_vectorized = -1
            AND is_archived = 0
            AND (metadata IS NULL 
                 OR json_extract(metadata, '$.vectorization_retries') IS NULL
                 OR json_extract(metadata, '$.vectorization_retries') < ?)
            ORDER BY created_time ASC
            LIMIT ?
        """, (max_retries, limit))
        return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def increment_vectorization_retry(self, record_id: int) -> int:
        """
        增加向量化重试计数
        
        Args:
            record_id: 记录ID
        
        Returns:
            当前重试次数
        """
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE memories 
                SET metadata = json_set(
                    COALESCE(metadata, '{}'),
                    '$.vectorization_retries',
                    COALESCE(json_extract(metadata, '$.vectorization_retries'), 0) + 1
                )
                WHERE id = ?
            """, (record_id,))
            conn.commit()
            
            cursor.execute("""
                SELECT json_extract(metadata, '$.vectorization_retries')
                FROM memories WHERE id = ?
            """, (record_id,))
            result = cursor.fetchone()
            return result[0] if result else 0
    
    def get_pending_compressions(self, limit: int = 20) -> List[MemoryRecord]:
        """获取待压缩的记录"""
        conn = self._get_read_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM memories 
            WHERE json_extract(metadata, '$.pending_compression') = 1
            AND is_archived = 0
            ORDER BY created_time ASC
            LIMIT ?
        """, (limit,))
        return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def get_by_vector_id(self, vector_id: str) -> Optional[MemoryRecord]:
        """根据ChromaDB ID获取SQLite记录"""
        conn = self._get_read_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM memories WHERE vector_id = ?
        """, (vector_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_record(row)
        return None
    
    def exists_by_text_hash(self, text_hash: str) -> Tuple[bool, Optional[int]]:
        """
        检查文本哈希是否已存在（幂等性检查）
        
        Returns:
            (exists, record_id): 是否存在，存在的记录ID
        """
        conn = self._get_read_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM memories 
            WHERE text_hash = ? AND is_archived = 0
            LIMIT 1
        """, (text_hash,))
        row = cursor.fetchone()
        if row:
            return True, row[0]
        return False, None
    
    def compute_text_hash(self, text: str) -> str:
        """计算文本哈希（用于幂等性检查）"""
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
    
    def add_with_hash(self, record: MemoryRecord, text_hash: str = None) -> int:
        """
        添加记忆记录（带文本哈希，支持幂等性）
        
        Returns:
            record_id: 记录ID（如果已存在则返回现有ID）
        """
        if text_hash is None:
            text_hash = self.compute_text_hash(record.text)
        
        exists, existing_id = self.exists_by_text_hash(text_hash)
        if exists:
            return existing_id
        
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories 
                (text, compressed_text, source, weight, access_count, 
                 last_access_time, created_time, metadata, vector_id, is_vectorized, text_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.text,
                record.compressed_text,
                record.source,
                record.weight,
                record.access_count,
                record.last_access_time,
                record.created_time,
                json.dumps(record.metadata, ensure_ascii=False),
                record.vector_id,
                record.is_vectorized,
                text_hash
            ))
            conn.commit()
            return cursor.lastrowid
    
    def add_batch_with_hash(self, records: List[MemoryRecord], text_hashes: List[str] = None) -> List[int]:
        """
        批量添加记忆记录（带文本哈希，支持幂等性）
        
        使用单条插入获取准确的 lastrowid，避免 MAX(id) 计算错误
        
        Args:
            records: 记录列表
            text_hashes: 文本哈希列表（可选，不提供则自动计算）
        
        Returns:
            记录ID列表
        """
        if not records:
            return []
        
        if text_hashes is None:
            text_hashes = [self.compute_text_hash(r.text) for r in records]
        
        existing_map = {}
        for text_hash in text_hashes:
            exists, existing_id = self.exists_by_text_hash(text_hash)
            if exists:
                existing_map[text_hash] = existing_id
        
        new_records = []
        new_hashes = []
        result_ids = []
        
        for record, text_hash in zip(records, text_hashes):
            if text_hash in existing_map:
                result_ids.append(existing_map[text_hash])
            else:
                new_records.append(record)
                new_hashes.append(text_hash)
        
        if new_records:
            with self._get_write_connection() as conn:
                cursor = conn.cursor()
                
                for record, text_hash in zip(new_records, new_hashes):
                    cursor.execute("""
                        INSERT INTO memories 
                        (text, compressed_text, source, weight, access_count, 
                         last_access_time, created_time, metadata, vector_id, is_vectorized, text_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.text,
                        record.compressed_text,
                        record.source,
                        record.weight,
                        record.access_count,
                        record.last_access_time,
                        record.created_time,
                        json.dumps(record.metadata, ensure_ascii=False),
                        record.vector_id,
                        record.is_vectorized,
                        text_hash
                    ))
                    result_ids.append(cursor.lastrowid)
                
                conn.commit()
        
        return result_ids
    
    def batch_update_vector_status(
        self, 
        updates: List[tuple], 
        is_vectorized: int = 1,
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        批量更新记录的向量化状态（事务保护版）
        
        Args:
            updates: [(record_id, vector_id), ...] 列表
            is_vectorized: 向量化状态
            batch_size: 分批提交大小（默认100）
        
        Returns:
            {"updated": 更新数量, "batches": 批次数}
        
        事务保护：
            - 使用 _get_write_connection 上下文管理器
            - 异常时自动回滚
            - 分批提交，避免单次事务过大
        """
        if not updates:
            return {"updated": 0, "batches": 0}
        
        total_updated = 0
        batch_count = 0
        
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            
            with self._get_write_connection() as conn:
                cursor = conn.cursor()
                
                data = [(vid, is_vectorized, rid) for rid, vid in batch]
                
                cursor.executemany("""
                    UPDATE memories 
                    SET vector_id = ?, is_vectorized = ?
                    WHERE id = ?
                """, data)
                
                conn.commit()
                total_updated += cursor.rowcount
                batch_count += 1
        
        return {"updated": total_updated, "batches": batch_count}
    
    def delete_by_vector_id(self, vector_id: str) -> bool:
        """根据ChromaDB ID删除SQLite记录（同步删除）"""
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM memories WHERE vector_id = ?
            """, (vector_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def __len__(self) -> int:
        conn = self._get_read_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories WHERE is_archived = 0")
        return cursor.fetchone()[0]
    
    def count(self) -> int:
        """获取记录总数"""
        return self.__len__()
    
    # ==================== 会话历史管理 ====================
    
    def add_conversation(
        self, 
        session_id: str, 
        user_input: str, 
        assistant_response: str,
        source: str = "local",
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        添加会话记录
        
        Args:
            session_id: 会话ID
            user_input: 用户输入
            assistant_response: 助理回复
            source: 来源 (local/cloud)
            metadata: 元数据
        
        Returns:
            记录ID
        """
        from datetime import datetime
        
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations 
                (session_id, user_input, assistant_response, source, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                user_input,
                assistant_response,
                source,
                datetime.now().isoformat(),
                json.dumps(metadata or {}, ensure_ascii=False)
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_conversations(
        self, 
        session_id: str = None, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        获取会话历史（分页）
        
        Args:
            session_id: 会话ID（可选，不指定则获取所有）
            limit: 返回数量
            offset: 偏移量
        
        Returns:
            会话记录列表
        """
        conn = self._get_read_connection()
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (session_id, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM conversations 
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "user_input": row["user_input"],
                "assistant_response": row["assistant_response"],
                "source": row["source"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            })
        
        return results
    
    def get_recent_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取最近的会话历史
        
        Args:
            limit: 返回数量
        
        Returns:
            会话记录列表（按时间正序）
        """
        conn = self._get_read_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM conversations 
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "user": row["user_input"],
                "assistant": row["assistant_response"],
                "source": row["source"],
                "timestamp": row["timestamp"]
            })
        
        return list(reversed(results))
    
    def clear_conversations(self, session_id: str = None):
        """
        清空会话历史
        
        Args:
            session_id: 会话ID（可选，不指定则清空所有）
        """
        with self._get_write_connection() as conn:
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            else:
                cursor.execute("DELETE FROM conversations")
            
            conn.commit()
    
    def get_conversation_count(self, session_id: str = None) -> int:
        """获取会话记录数量"""
        conn = self._get_read_connection()
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", (session_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM conversations")
        
        return cursor.fetchone()[0]
    
    def close(self):
        """
        关闭数据库连接
        
        用于会话结束或测试清理
        """
        if hasattr(self._local, 'read_conn') and self._local.read_conn is not None:
            try:
                self._local.read_conn.close()
            except Exception:
                pass
            self._local.read_conn = None
    
    def enqueue_pending(self, data: Dict[str, Any]) -> int:
        """
        将数据写入待处理队列
        
        当内存队列满时，将数据持久化到 SQLite 的 pending_queue 表。
        
        Args:
            data: 待处理的数据（通常是对话数据）
        
        Returns:
            插入的记录ID
        """
        def _do_enqueue():
            with self._get_write_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO pending_queue (data, created_time, status)
                    VALUES (?, ?, 'pending')
                """, (json.dumps(data, ensure_ascii=False), datetime.now().isoformat()))
                conn.commit()
                return cursor.lastrowid
        
        return self._retry_on_connection_error(_do_enqueue)
    
    def dequeue_pending(self, limit: int = 10) -> List[Tuple[int, Dict[str, Any]]]:
        """
        从待处理队列获取数据
        
        获取并标记为处理中的记录。
        
        Args:
            limit: 最大获取数量
        
        Returns:
            [(id, data), ...] 列表
        """
        def _do_dequeue():
            conn = self._get_read_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, data FROM pending_queue 
                WHERE status = 'pending'
                ORDER BY created_time ASC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[1])
                    results.append((row[0], data))
                except json.JSONDecodeError:
                    continue
            
            return results
        
        return self._retry_on_connection_error(_do_dequeue)
    
    def mark_pending_processed(self, record_ids: List[int]):
        """
        标记待处理记录为已完成
        
        Args:
            record_ids: 记录ID列表
        """
        if not record_ids:
            return
        
        def _do_mark():
            with self._get_write_connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' * len(record_ids))
                cursor.execute(f"""
                    DELETE FROM pending_queue 
                    WHERE id IN ({placeholders})
                """, record_ids)
                conn.commit()
        
        self._retry_on_connection_error(_do_mark)
    
    def get_pending_queue_count(self) -> int:
        """
        获取待处理队列中的记录数量
        
        Returns:
            待处理记录数量
        """
        def _do_count():
            conn = self._get_read_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pending_queue WHERE status = 'pending'")
            return cursor.fetchone()[0]
        
        return self._retry_on_connection_error(_do_count)
    
    def cleanup_old_pending_records(self, max_age_hours: int = 24):
        """
        清理过旧的待处理记录
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        def _do_cleanup():
            cutoff_time = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
            with self._get_write_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM pending_queue 
                    WHERE created_time < ? AND status = 'pending'
                """, (cutoff_time,))
                deleted = cursor.rowcount
                conn.commit()
                return deleted
        
        return self._retry_on_connection_error(_do_cleanup)


# 全局实例
_sqlite_store: Optional[SQLiteStore] = None
_instance_lock = threading.Lock()


def get_sqlite_store() -> SQLiteStore:
    """获取全局SQLite存储实例（线程安全单例）"""
    global _sqlite_store
    if _sqlite_store is None:
        with _instance_lock:
            if _sqlite_store is None:
                _sqlite_store = SQLiteStore(config.sqlite_db_path)
    return _sqlite_store


def reset_sqlite_store():
    """
    重置全局SQLite存储实例
    
    用于清空会话或重新初始化
    """
    global _sqlite_store
    with _instance_lock:
        if _sqlite_store is not None:
            _sqlite_store.close()
            _sqlite_store = None
