# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: config_manager.py
Modül: cognitive_management/v2/config
Görev: V2 Configuration Manager - Advanced configuration system with hot reload
       support. Environment-based configuration, config validation, config schema,
       hot reload support, config change notifications ve thread-safe operations
       sağlar. Akademik referans: Fowler, M. (2004).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (config yönetimi)
- Design Patterns: Manager Pattern (config management)
- Endüstri Standartları: Configuration management best practices

KULLANIM:
- Config yönetimi için
- Hot reload için
- Config validation için

BAĞIMLILIKLAR:
- Modül içi bağımlılıklar

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
import threading
import time
# Optional: File watching support
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError, ConfigError


@dataclass
class ConfigChangeEvent:
    """Config change event"""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float = field(default_factory=time.time)


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for config file changes"""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and event.src_path == self.config_manager.config_path:
            self.config_manager._reload_from_file()


class ConfigManager:
    """
    Advanced configuration manager.
    
    Features:
    - Environment-based config
    - Config validation
    - Hot reload
    - Change notifications
    - Thread-safe operations
    
    Akademik Referans:
    - Fowler, M. (2004). "Configuration Management Pattern"
    - Industry Standard: Hot Reload Pattern
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        watch: bool = True,
        default_config: Optional[CognitiveManagerConfig] = None
    ):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to config file (JSON or YAML)
            watch: Enable file watching for hot reload
            default_config: Default configuration (if file not found)
        """
        self.config_path = config_path
        self.watch = watch
        self.default_config = default_config or CognitiveManagerConfig()
        
        # Current config
        self._config: CognitiveManagerConfig = self.default_config
        
        # Config cache (for fast access)
        self._config_cache: Dict[str, Any] = {}
        
        # Change listeners
        self._listeners: List[Callable[[ConfigChangeEvent], None]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # File watcher
        self._observer: Optional[Observer] = None
        
        # Load initial config
        if config_path and os.path.exists(config_path):
            self._load_config()
        else:
            # Use default config
            self._config = self.default_config
            self._update_cache()
        
        # Start file watcher if enabled
        if watch and config_path:
            self._start_watcher()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.
        
        Args:
            key: Config key (e.g., "critic.enabled", "tools.enable_tools")
            default: Default value if key not found
            
        Returns:
            Config value
        """
        with self._lock:
            keys = key.split('.')
            value = self._config_cache
            
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                    if value is None:
                        return default
                else:
                    return default
            
            return value if value is not None else default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set config value (with hot reload).
        
        Args:
            key: Config key (dot notation)
            value: New value
        """
        with self._lock:
            old_value = self.get(key)
            
            # Update config object
            keys = key.split('.')
            config_dict = asdict(self._config)
            
            # Navigate to nested dict
            current = config_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set value
            current[keys[-1]] = value
            
            # Reconstruct config object
            try:
                self._config = CognitiveManagerConfig.from_dict(config_dict)
                self._config.validate()
            except Exception as e:
                raise ConfigError(f"Invalid config value for '{key}': {e}") from e
            
            # Update cache
            self._update_cache()
            
            # Save to file if path exists
            if self.config_path:
                self._save_to_file()
            
            # Notify listeners
            event = ConfigChangeEvent(
                key=key,
                old_value=old_value,
                new_value=value
            )
            self._notify_listeners(event)
    
    def reload(self) -> None:
        """
        Reload config from file.
        
        Raises:
            ConfigError: If config file is invalid
        """
        with self._lock:
            if not self.config_path or not os.path.exists(self.config_path):
                raise ConfigError(f"Config file not found: {self.config_path}")
            
            self._load_config()
    
    def register_listener(self, listener: Callable[[ConfigChangeEvent], None]) -> None:
        """
        Register config change listener.
        
        Args:
            listener: Callback function that receives ConfigChangeEvent
        """
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)
    
    def unregister_listener(self, listener: Callable[[ConfigChangeEvent], None]) -> None:
        """
        Unregister config change listener.
        
        Args:
            listener: Callback function to remove
        """
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
    
    def get_config(self) -> CognitiveManagerConfig:
        """
        Get current config object.
        
        Returns:
            Current CognitiveManagerConfig instance
        """
        with self._lock:
            return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update config with dictionary of changes.
        
        Args:
            updates: Dictionary of config updates (can be nested)
        """
        with self._lock:
            # Merge updates
            config_dict = asdict(self._config)
            self._deep_merge(config_dict, updates)
            
            # Reconstruct config
            try:
                self._config = CognitiveManagerConfig.from_dict(config_dict)
                self._config.validate()
            except Exception as e:
                raise ConfigError(f"Invalid config updates: {e}") from e
            
            # Update cache
            self._update_cache()
            
            # Save to file if path exists
            if self.config_path:
                self._save_to_file()
            
            # Notify listeners for all changed keys
            for key, value in self._flatten_dict(updates):
                old_value = self.get(key)
                event = ConfigChangeEvent(
                    key=key,
                    old_value=old_value,
                    new_value=value
                )
                self._notify_listeners(event)
    
    def validate_config(self) -> bool:
        """
        Validate current config.
        
        Returns:
            True if config is valid
            
        Raises:
            ConfigError: If config is invalid
        """
        with self._lock:
            try:
                self._config.validate()
                return True
            except Exception as e:
                raise ConfigError(f"Config validation failed: {e}") from e
    
    def export_config(self, path: Optional[str] = None) -> str:
        """
        Export config to JSON file.
        
        Args:
            path: Output path (None = use config_path)
            
        Returns:
            Exported config as JSON string
        """
        with self._lock:
            config_dict = asdict(self._config)
            json_str = json.dumps(config_dict, indent=2, ensure_ascii=False)
            
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
            
            return json_str
    
    def close(self) -> None:
        """Stop file watcher and cleanup"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
    
    def _load_config(self) -> None:
        """Load config from file"""
        if not self.config_path or not os.path.exists(self.config_path):
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.json'):
                    config_dict = json.load(f)
                elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    import yaml
                    config_dict = yaml.safe_load(f)
                else:
                    # Try JSON first
                    try:
                        config_dict = json.load(f)
                    except:
                        import yaml
                        f.seek(0)
                        config_dict = yaml.safe_load(f)
            
            # Merge with default config
            default_dict = asdict(self.default_config)
            self._deep_merge(default_dict, config_dict)
            
            # Reconstruct config
            self._config = CognitiveManagerConfig.from_dict(default_dict)
            self._config.validate()
            
            # Update cache
            self._update_cache()
            
        except Exception as e:
            raise ConfigError(f"Failed to load config from {self.config_path}: {e}") from e
    
    def _save_to_file(self) -> None:
        """Save config to file"""
        if not self.config_path:
            return
        
        try:
            config_dict = asdict(self._config)
            json_str = json.dumps(config_dict, indent=2, ensure_ascii=False)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(self.config_path) or '.', exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
                
        except Exception as e:
            raise ConfigError(f"Failed to save config to {self.config_path}: {e}") from e
    
    def _reload_from_file(self) -> None:
        """Reload config from file (called by file watcher)"""
        try:
            # Small delay to avoid reading while file is being written
            time.sleep(0.1)
            self.reload()
        except Exception as e:
            # Log error but don't crash
            print(f"Warning: Failed to reload config: {e}")
    
    def _start_watcher(self) -> None:
        """Start file system watcher"""
        if not self.config_path or not _WATCHDOG_AVAILABLE:
            return
        
        try:
            self._observer = Observer()
            handler = ConfigFileHandler(self)
            directory = os.path.dirname(self.config_path) or '.'
            self._observer.schedule(handler, directory, recursive=False)
            self._observer.start()
        except Exception as e:
            # File watching is optional, don't crash if it fails
            print(f"Warning: Failed to start config file watcher: {e}")
    
    def _update_cache(self) -> None:
        """Update config cache"""
        self._config_cache = asdict(self._config)
    
    def _notify_listeners(self, event: ConfigChangeEvent) -> None:
        """Notify all listeners of config change"""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                # Don't let listener errors crash the system
                print(f"Warning: Config change listener error: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep merge two dictionaries"""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> List[tuple]:
        """Flatten nested dictionary to list of (key, value) tuples"""
        items = []
        for key, value in d.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key))
            else:
                items.append((new_key, value))
        return items


# Add from_dict method to CognitiveManagerConfig if not exists
if not hasattr(CognitiveManagerConfig, 'from_dict'):
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CognitiveManagerConfig:
        """Create CognitiveManagerConfig from dictionary"""
        # This is a simplified version - in production, use proper dataclass reconstruction
        cfg = CognitiveManagerConfig()
        cfg.override_with(data)
        return cfg
    
    CognitiveManagerConfig.from_dict = from_dict


__all__ = ["ConfigManager", "ConfigChangeEvent"]

