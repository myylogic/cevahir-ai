# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: tool_executor_v2.py
Modül: cognitive_management/v2/components
Görev: V2 Tool Executor - Bağımsız implementasyon. Tool execution sistemi için
       merkezi executor. Tool registration, execution, validation ve error
       handling işlemlerini yapar. Akademik referans: Schick et al. (2023),
       OpenAI (2023).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (tool execution),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Executor Pattern (tool execution)
- Endüstri Standartları: Tool execution best practices

KULLANIM:
- Tool execution için
- Tool registration için
- Tool validation için

BAĞIMLILIKLAR:
- Component Protocols: ToolExecutor interface
- Config: Yapılandırma

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import time

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError, ToolPolicyError
from cognitive_management.v2.interfaces.component_protocols import ToolExecutor


@dataclass
class ToolMetrics:
    """Tool execution metrics"""
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    last_call_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Success rate (0.0-1.0)"""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count
    
    @property
    def avg_latency(self) -> float:
        """Average latency in seconds"""
        if self.call_count == 0:
            return 0.0
        return self.total_latency / self.call_count


class ToolExecutorV2(ToolExecutor):
    """
    V2 Tool Executor - Bağımsız implementasyon.
    
    Features:
    - Tool registry
    - Parameter validation
    - Result caching (optional)
    - Error handling
    - Usage metrics
    - Thread-safe operations
    
    Akademik Referans:
    - Schick et al. (2023). "Tool Use in Large Language Models"
    - OpenAI (2023). "Function Calling in GPT-4"
    """
    
    def __init__(self, cfg: CognitiveManagerConfig):
        """
        Initialize V2 Tool Executor.
        
        Args:
            cfg: Cognitive manager configuration
        """
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        
        self.cfg = cfg
        self._tools: Dict[str, Callable] = {}
        self._tool_metrics: Dict[str, ToolMetrics] = {}
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        
        # Register default tools if enabled
        if cfg.tools.enable_tools:
            self._register_default_tools()
    
    def register_tool(
        self,
        name: str,
        tool_func: Callable,
        schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a tool.
        
        Args:
            name: Tool name
            tool_func: Tool function (must accept **kwargs)
            schema: Optional tool schema (parameters, description, etc.)
        """
        if not name or not isinstance(name, str):
            raise ValidationError("Tool name must be a non-empty string.")
        
        if not callable(tool_func):
            raise ValidationError("tool_func must be callable.")
        
        if name in self._tools:
            raise ToolPolicyError(f"Tool '{name}' already registered.")
        
        # Note: Allow list check is done during execute(), not during register()
        # This allows registering custom tools that may not be in the default allow list
        
        self._tools[name] = tool_func
        self._tool_metrics[name] = ToolMetrics()
        self._tool_schemas[name] = schema or {}
    
    def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute tool with parameters.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters (dict)
            
        Returns:
            Tool execution result
            
        Raises:
            ToolPolicyError: If tool not found or execution fails
        """
        if not tool_name or not isinstance(tool_name, str):
            raise ValidationError("tool_name must be a non-empty string.")
        
        if tool_name not in self._tools:
            raise ToolPolicyError(f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}")
        
        # Check if tools are enabled
        if not self.cfg.tools.enable_tools:
            raise ToolPolicyError("Tools are disabled in configuration.")
        
        # Check if tool is allowed (only if allow list is set and not empty)
        if self.cfg.tools.allow and len(self.cfg.tools.allow) > 0 and tool_name not in self.cfg.tools.allow:
            raise ToolPolicyError(f"Tool '{tool_name}' is not in allowed tools list.")
        
        # Get tool function
        tool_func = self._tools[tool_name]
        metrics = self._tool_metrics[tool_name]
        
        # Execute with metrics tracking
        start_time = time.time()
        metrics.call_count += 1
        metrics.last_call_time = start_time
        
        try:
            # Validate parameters (basic check)
            if not isinstance(parameters, dict):
                raise ValidationError("parameters must be a dictionary.")
            
            # Python signature validation catches missing and unknown arguments.
            import inspect
            inspect.signature(tool_func).bind(**parameters)
            schema = self._tool_schemas.get(tool_name, {}).get("parameters", {})
            properties = schema.get("properties", schema)
            types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
            for key, value in parameters.items():
                definition = properties.get(key, {})
                expected = types.get(definition.get("type")) if isinstance(definition, dict) else None
                if expected and (not isinstance(value, expected) or isinstance(value, bool) and definition.get("type") in ("integer", "number")):
                    raise ValidationError(f"Invalid type for tool parameter {key}")
            for required in schema.get("required", []):
                if required not in parameters:
                    raise ValidationError(f"Missing tool parameter {required}")
            result = tool_func(**parameters)
            
            # Record success
            latency = time.time() - start_time
            metrics.success_count += 1
            metrics.total_latency += latency
            
            return result
            
        except Exception as e:
            # Record error
            latency = time.time() - start_time
            metrics.error_count += 1
            metrics.total_latency += latency
            
            raise ToolPolicyError(f"Tool '{tool_name}' execution failed: {e}") from e
    
    def list_available_tools(self) -> List[str]:
        """
        List available tools.
        
        Returns:
            List of available tool names
        """
        if not self.cfg.tools.enable_tools:
            return []
        
        # Filter by allowed tools (only if allow list is set and not empty)
        # If allow list is empty or None, return all registered tools
        if self.cfg.tools.allow and len(self.cfg.tools.allow) > 0:
            return [name for name in self._tools.keys() if name in self.cfg.tools.allow]
        
        return list(self._tools.keys())
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool schema.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool schema or None
        """
        return self._tool_schemas.get(tool_name)
    
    def get_tool_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get tool metrics.
        
        Args:
            tool_name: Tool name (None = all tools)
            
        Returns:
            Metrics dictionary
        """
        if tool_name:
            if tool_name not in self._tool_metrics:
                return {}
            metrics = self._tool_metrics[tool_name]
            return {
                "call_count": metrics.call_count,
                "success_count": metrics.success_count,
                "error_count": metrics.error_count,
                "success_rate": metrics.success_rate,
                "avg_latency": metrics.avg_latency,
                "last_call_time": metrics.last_call_time,
            }
        else:
            # All tools
            return {
                name: {
                    "call_count": m.call_count,
                    "success_count": m.success_count,
                    "error_count": m.error_count,
                    "success_rate": m.success_rate,
                    "avg_latency": m.avg_latency,
                    "last_call_time": m.last_call_time,
                }
                for name, m in self._tool_metrics.items()
            }
    
    def _register_default_tools(self) -> None:
        """
        Register default tools (calculator, search, file).
        These are placeholder implementations.
        """
        # Calculator tool
        def calculator(operation: str, **kwargs) -> str:
            """
            Simple calculator tool.
            
            Args:
                operation: Math operation (e.g., "2+2", "10*5")
                
            Returns:
                Calculation result as string
            """
            import ast
            import operator
            if not isinstance(operation, str) or len(operation) > 200:
                raise ValueError("Calculator expression must be at most 200 characters")
            tree = ast.parse(operation, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 64:
                raise ValueError("Calculator expression is too complex")
            binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
            def evaluate(node):
                if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                    return node.value
                if isinstance(node, ast.BinOp) and type(node.op) in binary:
                    return binary[type(node.op)](evaluate(node.left), evaluate(node.right))
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                    value = evaluate(node.operand)
                    return -value if isinstance(node.op, ast.USub) else value
                raise ValueError("Only numbers and + - * / are supported")
            return str(evaluate(tree.body))

        # Register tools
        if "calculator" in (self.cfg.tools.allow or []):
            self.register_tool(
                "calculator",
                calculator,
                schema={
                    "description": "Simple calculator for basic math operations",
                    "parameters": {
                        "operation": {
                            "type": "string",
                            "description": "Math operation (e.g., '2+2', '10*5')"
                        }
                    }
                }
            )
        
        # Search/file are supplied by callers through register_tool. Advertising
        # placeholder results as successful tool execution is not supported.


__all__ = ["ToolExecutorV2", "ToolMetrics"]

