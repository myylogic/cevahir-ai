# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: tool_policy_v2.py
Modül: cognitive_management/v2/components
Görev: V2 Tool Policy - Bağımsız implementasyon. Tool selection and usage policy.
       Tool selection, usage policy, validation ve decision işlemlerini yapar.
       Akademik referans: Schick et al. (2023), OpenAI (2023).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (tool policy),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Policy Pattern (tool policy)
- Endüstri Standartları: Tool policy best practices

KULLANIM:
- Tool selection için
- Tool usage policy için
- Tool validation için

BAĞIMLILIKLAR:
- Heuristics: Heuristic fonksiyonlar
- Component Protocols: ToolExecutor interface

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Optional

from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError
from cognitive_management.v2.utils.heuristics import should_tool
from cognitive_management.v2.interfaces.component_protocols import ToolExecutor


class ToolPolicyV2:
    """
    V2 Tool Policy - Bağımsız implementasyon.
    
    Features:
    - Tool selection based on features
    - Tool parameter inference
    - Tool usage policy (when to use tools)
    - Integration with ToolExecutor
    
    Akademik Referans:
    - Schick et al. (2023). "Tool Use in Large Language Models"
    - OpenAI (2023). "Function Calling in GPT-4"
    """
    
    def __init__(
        self,
        cfg: CognitiveManagerConfig,
        tool_executor: Optional[ToolExecutor] = None
    ):
        """
        Initialize V2 Tool Policy.
        
        Args:
            cfg: Cognitive manager configuration
            tool_executor: Optional tool executor (for tool availability check)
        """
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        
        self.cfg = cfg
        self.tool_executor = tool_executor
    
    def choose_tool(
        self,
        features: Dict[str, Any],
        wish: Optional[str] = None
    ) -> Optional[str]:
        """
        Choose tool based on features and policy.
        
        Args:
            features: Extracted features from input
            wish: Optional tool name preference (from policy router)
            
        Returns:
            Selected tool name or None
        """
        # Check if tools are enabled
        if not self.cfg.tools.enable_tools:
            return None
        
        # If wish is provided and valid, use it
        if wish and wish != "none":
            if self._is_tool_available(wish):
                return wish
            # Fallback to heuristic if wish is not available
        
        # Use heuristic to determine tool need
        tool_decision = should_tool(self.cfg, features)
        
        if tool_decision == "none":
            return None
        
        # Select tool based on features
        selected_tool = self._select_tool_from_features(features, tool_decision)
        
        return selected_tool
    
    def _is_tool_available(self, tool_name: str) -> bool:
        """
        Check if tool is available.
        
        Args:
            tool_name: Tool name
            
        Returns:
            True if tool is available
        """
        if not self.tool_executor:
            return False
        
        available_tools = self.tool_executor.list_available_tools()
        return tool_name in available_tools
    
    def _select_tool_from_features(
        self,
        features: Dict[str, Any],
        tool_decision: str
    ) -> Optional[str]:
        """
        Select tool from features.
        
        Args:
            features: Extracted features
            tool_decision: Tool decision ("maybe" or "must")
            
        Returns:
            Selected tool name or None
        """
        # Priority order based on config
        tool_priority = list(self.cfg.tools.allow) if self.cfg.tools.allow else []
        
        # If no priority, use default
        if not tool_priority:
            tool_priority = ["search", "calculator", "file"]
        
        # Check features to determine which tool
        needs_recent_info = features.get("needs_recent_info", False)
        needs_calc_or_parse = features.get("needs_calc_or_parse", False)
        
        # Search tool for recent info
        if needs_recent_info and "search" in tool_priority:
            if self._is_tool_available("search"):
                return "search"
        
        # Calculator tool for calculations
        if needs_calc_or_parse and "calculator" in tool_priority:
            if self._is_tool_available("calculator"):
                return "calculator"
        
        # If decision is "must", try to find any available tool
        if tool_decision == "must":
            for tool_name in tool_priority:
                if self._is_tool_available(tool_name):
                    return tool_name
        
        # If decision is "maybe", be more conservative
        if tool_decision == "maybe":
            # Only use if there's a clear signal
            if needs_recent_info or needs_calc_or_parse:
                for tool_name in tool_priority:
                    if self._is_tool_available(tool_name):
                        return tool_name
        
        return None
    
    def infer_tool_parameters(
        self,
        tool_name: str,
        user_message: str,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Infer tool parameters from user message and features.
        
        Args:
            tool_name: Tool name
            user_message: User message
            features: Extracted features
            
        Returns:
            Tool parameters dictionary
        """
        parameters: Dict[str, Any] = {}
        
        if tool_name == "calculator":
            expression = self._infer_calculator_expression(user_message)
            if expression is not None:
                parameters["operation"] = expression
        
        elif tool_name == "search":
            # Extract search query from user message
            # Remove common words and use the rest as query
            stop_words = {"ne", "nedir", "nasıl", "hangi", "nerede", "kim", "ne zaman"}
            words = user_message.lower().split()
            query_words = [w for w in words if w not in stop_words]
            if query_words:
                parameters["query"] = " ".join(query_words[:10])  # Limit to 10 words
            else:
                parameters["query"] = user_message[:100]  # Fallback
        
        elif tool_name == "file":
            # Try to extract file path from user message
            import re
            # Look for file paths (simple pattern)
            path_pattern = r'[\w/\\]+\.\w+'
            matches = re.findall(path_pattern, user_message)
            if matches:
                parameters["path"] = matches[0]
            else:
                # Fallback: use message as path (not ideal, but better than nothing)
                parameters["path"] = user_message.strip()[:200]
        
        return parameters

    @staticmethod
    def _infer_calculator_expression(message: str) -> Optional[str]:
        """Keep one complete arithmetic span; never invent an operation.

        This is deliberately a small expression recognizer, not natural-language
        math understanding. Unsupported or ambiguous spans require explicit
        parameters. Parsing validates syntax without executing user input.
        """
        import ast
        import re

        candidates = []
        # Include unsupported mathematical punctuation so that e.g. 2^3+4
        # cannot silently become the valid but different expression 3+4.
        math_char = r"[0-9.+\-*/()%<>=^&|~!,\[\]{}\\]"
        pattern = math_char + r"(?:" + math_char + r"|[eE][+\-]?[0-9]+|\s+(?=" + math_char + r"))*"
        for match in re.finditer(pattern, message):
            expression = match.group().strip()
            if not any(c.isdigit() for c in expression) or not any(c in "+-*/%^" for c in expression):
                continue
            if ((match.start() and message[match.start()-1].isalnum()) or
                    (match.end() < len(message) and message[match.end()].isalnum())):
                return None
            if len(expression) > 200:
                return None
            try:
                tree = ast.parse(expression, mode="eval")
            except (SyntaxError, ValueError, RecursionError):
                return None
            nodes = list(ast.walk(tree))
            allowed = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp,
                       ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub)
            if (len(nodes) > 64 or not any(isinstance(n, ast.BinOp) for n in nodes) or
                    any(not isinstance(n, allowed) for n in nodes) or
                    any(isinstance(n, ast.Constant) and type(n.value) not in (int, float) for n in nodes)):
                return None
            candidates.append(expression)
        return candidates[0] if len(candidates) == 1 else None


__all__ = ["ToolPolicyV2"]

