# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: connection_pool.py
Modül: cognitive_management/v2/utils
Görev: Connection Pool - Connection pooling for efficient resource management.
       Phase 6: Performance Optimization & Caching Enhancement. Connection protocol,
       ConnectionPool, PooledConnection sınıflarını içerir. Connection pooling,
       connection lifecycle management ve efficient resource management işlemlerini
       yapar. Akademik referans: Object Pool Pattern (Design Patterns, Gang of Four),
       Connection Pooling Best Practices.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (connection pooling)
- Design Patterns: Object Pool Pattern (connection pooling)
- Endüstri Standartları: Connection pooling best practices

KULLANIM:
- Connection pooling için
- Resource management için
- Connection lifecycle management için

BAĞIMLILIKLAR:
- threading: Thread-safe işlemler
- typing: Generic type hints

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Protocol, Optional, Generic, TypeVar, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time
from collections import deque

T = TypeVar('T')


class Connection(Protocol):
    """Connection protocol - must implement close() method."""
    def close(self) -> None: ...


@dataclass
class PooledConnection:
    """
    Pooled connection wrapper.
    
    Attributes:
        connection: The actual connection object
        created_at: When connection was created
        last_used: When connection was last used
        use_count: Number of times connection has been used
        is_active: Whether connection is currently in use
    """
    connection: Any
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    is_active: bool = False


class ConnectionPool(Generic[T]):
    """
    Generic connection pool implementation.
    
    Features:
    - Connection reuse
    - Maximum pool size
    - Connection lifecycle management
    - Thread-safe
    - Connection health checks
    """
    
    def __init__(
        self,
        factory: callable,
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: float = 300.0,  # 5 minutes
        connection_timeout: float = 30.0,
        health_check: Optional[callable] = None,
    ):
        """
        Initialize connection pool.
        
        Args:
            factory: Function to create new connections
            max_size: Maximum pool size
            min_size: Minimum pool size (keep this many connections alive)
            max_idle_time: Maximum idle time before connection is closed (seconds)
            connection_timeout: Timeout for getting connection from pool (seconds)
            health_check: Optional function to check connection health (returns bool)
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.connection_timeout = connection_timeout
        self.health_check = health_check
        
        # Pool storage
        self._available: deque = deque()
        self._in_use: Dict[int, PooledConnection] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Statistics
        self._total_created = 0
        self._total_closed = 0
        self._total_checked_out = 0
        self._total_checked_in = 0
    
    def acquire(self) -> T:
        """
        Acquire connection from pool.
        
        Returns:
            Connection object
            
        Raises:
            TimeoutError: If connection cannot be acquired within timeout
        """
        with self._condition:
            deadline = time.time() + self.connection_timeout
            
            while True:
                # Check if timeout expired
                if time.time() > deadline:
                    raise TimeoutError(f"Connection pool timeout after {self.connection_timeout}s")
                
                # Try to get available connection
                connection_wrapper = self._get_available_connection()
                
                if connection_wrapper:
                    # Mark as in use
                    connection_wrapper.is_active = True
                    connection_wrapper.last_used = datetime.now()
                    connection_wrapper.use_count += 1
                    
                    # Move to in-use dict
                    conn_id = id(connection_wrapper.connection)
                    self._in_use[conn_id] = connection_wrapper
                    
                    self._total_checked_out += 1
                    return connection_wrapper.connection
                
                # No available connection - try to create new one
                if len(self._available) + len(self._in_use) < self.max_size:
                    # Create new connection
                    new_conn = self.factory()
                    new_wrapper = PooledConnection(
                        connection=new_conn,
                        created_at=datetime.now(),
                        last_used=datetime.now(),
                        use_count=1,
                        is_active=True,
                    )
                    
                    conn_id = id(new_conn)
                    self._in_use[conn_id] = new_wrapper
                    self._total_created += 1
                    self._total_checked_out += 1
                    
                    return new_conn
                
                # Pool is full - wait for connection to be released
                remaining_time = deadline - time.time()
                if remaining_time > 0:
                    self._condition.wait(timeout=remaining_time)
                else:
                    raise TimeoutError(f"Connection pool timeout after {self.connection_timeout}s")
    
    def release(self, connection: T) -> None:
        """
        Release connection back to pool.
        
        Args:
            connection: Connection object to release
        """
        with self._lock:
            conn_id = id(connection)
            wrapper = self._in_use.pop(conn_id, None)
            
            if not wrapper:
                # Connection not from this pool - close it
                if hasattr(connection, 'close'):
                    connection.close()
                return
            
            # Check if connection is still healthy
            if self.health_check and not self.health_check(connection):
                # Connection is unhealthy - close it
                if hasattr(connection, 'close'):
                    connection.close()
                self._total_closed += 1
                return
            
            # Mark as not in use
            wrapper.is_active = False
            wrapper.last_used = datetime.now()
            
            # Return to available pool
            self._available.append(wrapper)
            self._total_checked_in += 1
            
            # Notify waiting threads
            self._condition.notify()
    
    def _get_available_connection(self) -> Optional[PooledConnection]:
        """Get available connection from pool, cleaning up idle connections."""
        while self._available:
            wrapper = self._available.popleft()
            
            # Check if connection is idle too long
            idle_time = (datetime.now() - wrapper.last_used).total_seconds()
            if idle_time > self.max_idle_time:
                # Close idle connection
                if hasattr(wrapper.connection, 'close'):
                    wrapper.connection.close()
                self._total_closed += 1
                continue
            
            # Check health if health check available
            if self.health_check and not self.health_check(wrapper.connection):
                # Unhealthy connection - close it
                if hasattr(wrapper.connection, 'close'):
                    wrapper.connection.close()
                self._total_closed += 1
                continue
            
            return wrapper
        
        return None
    
    def close_all(self) -> None:
        """Close all connections in pool."""
        with self._lock:
            # Close available connections
            while self._available:
                wrapper = self._available.popleft()
                if hasattr(wrapper.connection, 'close'):
                    wrapper.connection.close()
                self._total_closed += 1
            
            # Close in-use connections
            for wrapper in list(self._in_use.values()):
                if hasattr(wrapper.connection, 'close'):
                    wrapper.connection.close()
                self._total_closed += 1
            
            self._available.clear()
            self._in_use.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "available": len(self._available),
                "in_use": len(self._in_use),
                "total_created": self._total_created,
                "total_closed": self._total_closed,
                "total_checked_out": self._total_checked_out,
                "total_checked_in": self._total_checked_in,
                "max_size": self.max_size,
                "min_size": self.min_size,
            }
    
    def cleanup_idle(self) -> int:
        """
        Clean up idle connections.
        
        Returns:
            Number of connections cleaned up
        """
        with self._lock:
            cleaned = 0
            
            # Check available connections
            remaining = []
            for wrapper in self._available:
                idle_time = (datetime.now() - wrapper.last_used).total_seconds()
                
                # Keep if within min_size or not idle too long
                if len(self._available) + len(self._in_use) <= self.min_size:
                    remaining.append(wrapper)
                elif idle_time > self.max_idle_time:
                    # Close idle connection
                    if hasattr(wrapper.connection, 'close'):
                        wrapper.connection.close()
                    self._total_closed += 1
                    cleaned += 1
                else:
                    remaining.append(wrapper)
            
            self._available = deque(remaining)
            return cleaned


__all__ = [
    "ConnectionPool",
    "PooledConnection",
    "Connection",
]

