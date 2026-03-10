# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: component_protocols.py
Modül: cognitive_management/v2/interfaces
Görev: Component Protocol Definitions - Cognitive management component'leri için
       interface tanımları. PolicyRouter, MemoryService, DeliberationEngine,
       Critic, ToolExecutor ve diğer component interface tanımlarını içerir.
       SOLID prensipleri ile interface segregation sağlar.

MİMARİ:
- SOLID Prensipleri: Interface Segregation Principle (ISP),
                     Dependency Inversion Principle (DIP)
- Design Patterns: Protocol Pattern (component protocols)
- Endüstri Standartları: Interface design best practices

KULLANIM:
- Component interface tanımları için
- Cognitive management protocols için
- Interface segregation için

BAĞIMLILIKLAR:
- typing.Protocol: Protocol tanımları
- CognitiveTypes: Tip tanımları

Yazar: Muhammed Yasin Yılmaz
Telif Hakkı: © 2024 Muhammed Yasin Yılmaz. Tüm Hakları Saklıdır.
Kullanım: Bu dosya Cevahir-AI projesinin bir parçasıdır.
          İzinsiz kullanım, kopyalama, dağıtım veya değiştirme yasaktır.
          Ticari veya ticari olmayan herhangi bir amaçla kullanım için
          yazılı izin gereklidir.

================================================================================
"""

from __future__ import annotations
from typing import Protocol, Dict, Any, Optional, List, Tuple

# V1'den import (backward compatibility)
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.cognitive_types import (
    CognitiveState,
    CognitiveInput,
    CognitiveOutput,
    PolicyOutput,
    DecodingConfig,
    ThoughtCandidate,
)


# =============================================================================
# Component Interfaces
# =============================================================================

class PolicyRouter(Protocol):
    """
    Policy routing interface.
    Strateji seçimi için.
    """
    def route(
        self,
        features: Dict[str, Any],
        state: CognitiveState
    ) -> PolicyOutput:
        """
        Route to appropriate policy based on features.
        
        Args:
            features: Extracted features from input
            state: Current cognitive state
            
        Returns:
            PolicyOutput with selected strategy
        """
        ...


class MemoryService(Protocol):
    """
    Memory service interface.
    Conversation history ve context management için.
    """
    def add_turn(
        self,
        history: List[Dict[str, Any]],
        role: str,
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Add conversation turn to history.
        
        Args:
            history: Current conversation history
            role: Role ("user" or "assistant")
            content: Message content
            
        Returns:
            Updated history
        """
        ...
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from memory.
        
        Args:
            query: Query text
            top_k: Number of relevant items to retrieve
            
        Returns:
            List of relevant context items
        """
        ...
    
    def summarize(
        self,
        history: List[Dict[str, Any]],
        max_length: int = 200
    ) -> str:
        """
        Summarize conversation history.
        
        Args:
            history: Conversation history
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        ...


class Critic(Protocol):
    """
    Critic interface.
    Output quality control için.
    """
    def review(
        self,
        user_message: str,
        draft_text: str,
        context: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Review and revise text if needed.
        
        Args:
            user_message: Original user message
            draft_text: Draft response text
            context: Optional context
            
        Returns:
            Tuple of (final_text, was_revised)
        """
        ...


class DeliberationEngine(Protocol):
    """
    Deliberation engine interface.
    Internal thought generation için.
    """
    def generate_thoughts(
        self,
        prompt: str,
        num_thoughts: int = 1,
        decoding_config: Optional[DecodingConfig] = None
    ) -> List[ThoughtCandidate]:
        """
        Generate internal thoughts.
        
        Args:
            prompt: Input prompt
            num_thoughts: Number of thoughts to generate
            decoding_config: Optional decoding configuration
            
        Returns:
            List of thought candidates
        """
        ...


class ToolExecutor(Protocol):
    """
    Tool execution interface.
    External tool usage için.
    """
    def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute external tool.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        ...
    
    def list_available_tools(self) -> List[str]:
        """
        List available tools.
        
        Returns:
            List of available tool names
        """
        ...


__all__ = [
    "PolicyRouter",
    "MemoryService",
    "Critic",
    "DeliberationEngine",
    "ToolExecutor",
]

