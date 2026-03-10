# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: handlers.py
Modül: cognitive_management/v2/processing
Görev: Processing Handlers - Chain of Responsibility pattern handler implementations.
       Her handler tek bir sorumluluğa sahip (SRP). FeatureExtractionHandler,
       PolicyRoutingHandler, DeliberationHandler, ContextBuildingHandler,
       GenerationHandler, CriticHandler ve diğer handler sınıflarını içerir.
       Request processing için handler chain sağlar.

MİMARİ:
- SOLID Prensipleri: Single Responsibility (her handler tek sorumluluk),
                     Chain of Responsibility Pattern
- Design Patterns: Chain of Responsibility Pattern (processing handlers)
- Endüstri Standartları: Handler pattern best practices

KULLANIM:
- Request processing handlers için
- Feature extraction için
- Policy routing için
- Deliberation için

BAĞIMLILIKLAR:
- ProcessingPipeline: Processing pipeline
- Heuristics: Heuristic fonksiyonlar
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
from typing import Optional

# V1'den import
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cognitive_management.cognitive_types import DecodingConfig, PolicyOutput, ThoughtCandidate
from cognitive_management.v2.utils.heuristics import build_features

from .pipeline import BaseProcessingHandler, ProcessingContext

# Cache support (optional)
try:
    from ..utils.cache import InMemoryCache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False


# =============================================================================
# Feature Extraction Handler
# =============================================================================

class FeatureExtractionHandler(BaseProcessingHandler):
    """
    Feature extraction handler.
    SOLID: SRP - Sadece feature extraction yapar.
    """
    
    def __init__(self, memory_service):
        super().__init__("FeatureExtraction")
        self.memory_service = memory_service
    
    def _process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Extract features from input.
        
        Phase 3: Enhanced with memory retrieval and advanced feature extraction.
        """
        from cognitive_management.config import CognitiveManagerConfig
        
        # Get config from memory service (if available)
        cfg = getattr(self.memory_service, 'cfg', CognitiveManagerConfig())
        
        # Entropy estimation (can be enhanced with backend)
        try:
            # Simple heuristic: question marks, uncertainty markers
            user_msg = context.request.user_message
            question_count = user_msg.count("?")
            uncertainty_markers = ["belki", "muhtemelen", "sanırım", "olabilir", "maybe", "perhaps"]
            uncertainty_count = sum(1 for marker in uncertainty_markers if marker.lower() in user_msg.lower())
            
            # Entropy: questions + uncertainty markers
            entropy = min(3.0, (question_count * 0.5) + (uncertainty_count * 0.3))
        except:
            entropy = 0.8
        
        # Memory retrieval: Get relevant context from episodic memory
        try:
            relevant_contexts = self.memory_service.retrieve_context(
                query=context.request.user_message,
                top_k=3
            )
            # Add retrieved contexts to features for context-aware routing
            context.retrieved_contexts = relevant_contexts
        except Exception:
            context.retrieved_contexts = []
        
        # Build features
        features = build_features(
            cfg,
            user_message=context.request.user_message,
            entropy_est=entropy,
        )
        
        # Enhance features with retrieved context info
        if context.retrieved_contexts:
            features["has_relevant_memory"] = True
            features["memory_relevance_score"] = max(
                (ctx.get("score", 0.0) for ctx in context.retrieved_contexts),
                default=0.0
            )
        else:
            features["has_relevant_memory"] = False
            features["memory_relevance_score"] = 0.0
        
        context.features = features
        return context


# =============================================================================
# Policy Routing Handler
# =============================================================================

class PolicyRoutingHandler(BaseProcessingHandler):
    """
    Policy routing handler.
    SOLID: SRP - Sadece policy routing yapar.
    """
    
    def __init__(self, policy_router):
        super().__init__("PolicyRouting")
        self.policy_router = policy_router
    
    def _process(self, context: ProcessingContext) -> ProcessingContext:
        """Route to appropriate policy"""
        # Route based on features
        policy_output = self.policy_router.route(
            features=context.features,
            state=context.state,
        )
        
        # Override decoding if provided
        if context.decoding_config:
            policy_output.decoding = context.decoding_config
        
        context.policy_output = policy_output
        return context


# =============================================================================
# Deliberation Handler
# =============================================================================

class DeliberationHandler(BaseProcessingHandler):
    """
    Deliberation handler.
    SOLID: SRP - Sadece deliberation yapar.
    
    Phase 4: Enhanced with Tree of Thoughts (ToT) support.
    """
    
    def __init__(self, engine, backend, cfg=None, tree_of_thoughts=None):
        super().__init__("Deliberation")
        self.engine = engine
        self.backend = backend
        self.cfg = cfg
        self.tree_of_thoughts = tree_of_thoughts  # Phase 4: ToT support
    
    def _process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Generate internal thoughts if needed.
        
        Phase 4: Supports think1, debate2, and tot modes.
        """
        policy = context.policy_output
        
        if not policy or policy.mode not in ("think1", "debate2", "tot"):
            return context
        
        # Phase 4: Tree of Thoughts mode
        if policy.mode == "tot":
            return self._process_tot(context, policy)
        
        # Original CoT/Debate logic
        try:
            n = 1 if policy.mode == "think1" else 2
            thoughts = self.engine.generate_thoughts(
                prompt=context.request.user_message,
                num_thoughts=n,
                decoding_config=policy.decoding,
            )
            
            if thoughts:
                # Phase 8: Enhanced thought selection
                from cognitive_management.v2.utils.selectors import (
                    pick_best_by_score,
                    diversify_topk,
                    select_topk,
                )
                
                if policy.mode == "debate2" and len(thoughts) >= 2:
                    # For debate mode, use diverse selection to get different perspectives
                    diverse_thoughts = diversify_topk(thoughts, k=2, jaccard_threshold=0.7)
                    if diverse_thoughts:
                        # Pick best from diverse thoughts
                        context.selected_thought = pick_best_by_score(diverse_thoughts)
                    else:
                        # Fallback to regular selection
                        context.selected_thought = pick_best_by_score(thoughts)
                elif len(thoughts) > 1:
                    # For multiple thoughts, use top-k selection then pick best
                    top_thoughts = select_topk(thoughts, k=min(3, len(thoughts)))
                    context.selected_thought = pick_best_by_score(top_thoughts)
                else:
                    # Single thought or think mode - use simple selection
                    context.selected_thought = pick_best_by_score(thoughts)
        except Exception:
            # Deliberation hatası ana akışı düşürmesin
            context.selected_thought = None
        
        return context
    
    def _process_tot(self, context: ProcessingContext, policy) -> ProcessingContext:
        """
        Process Tree of Thoughts reasoning.
        
        Phase 4: Uses TreeOfThoughts component for complex problem solving.
        """
        if not self.tree_of_thoughts:
            # ToT not initialized - fallback to think mode
            import logging
            logging.warning("TreeOfThoughts not initialized, falling back to think mode")
            try:
                thoughts = self.engine.generate_thoughts(
                    prompt=context.request.user_message,
                    num_thoughts=1,
                    decoding_config=policy.decoding,
                )
                if thoughts:
                    from cognitive_management.v2.utils.selectors import pick_best_by_score
                    context.selected_thought = pick_best_by_score(thoughts)
            except Exception:
                context.selected_thought = None
            return context
        
        try:
            # Use Tree of Thoughts to solve the problem
            problem = context.request.user_message
            system_prompt = context.request.system_prompt or (
                self.cfg.default_system_prompt if self.cfg else None
            )
            
            # Solve using ToT
            best_paths = self.tree_of_thoughts.solve(
                problem=problem,
                system_prompt=system_prompt,
                decoding_config=policy.decoding,
            )
            
            if best_paths:
                # Get best path (first one)
                best_path, path_score = best_paths[0]
                
                # Combine path thoughts into a single thought
                # Last thought in path is typically the conclusion
                combined_thought = " → ".join(best_path[-3:])  # Last 3 steps
                
                # Create ThoughtCandidate from ToT result
                from cognitive_management.cognitive_types import ThoughtCandidate
                context.selected_thought = ThoughtCandidate(
                    text=combined_thought,
                    score=path_score,
                )
        except Exception as e:
            # ToT error - fallback to think mode
            import logging
            logging.warning(f"TreeOfThoughts error, falling back to think mode: {e}")
            try:
                thoughts = self.engine.generate_thoughts(
                    prompt=context.request.user_message,
                    num_thoughts=1,
                    decoding_config=policy.decoding,
                )
                if thoughts:
                    from cognitive_management.v2.utils.selectors import pick_best_by_score
                    context.selected_thought = pick_best_by_score(thoughts)
            except Exception:
                context.selected_thought = None
        
        return context


# =============================================================================
# Context Building Handler
# =============================================================================

class ContextBuildingHandler(BaseProcessingHandler):
    """
    Context building handler.
    SOLID: SRP - Sadece context building yapar.
    
    Phase 7.1: Enhanced with RAG (Retrieval-Augmented Generation).
    """
    
    def __init__(self, memory_service, tool_policy=None):
        super().__init__("ContextBuilding")
        self.memory_service = memory_service
        self.tool_policy = tool_policy
        # Phase 7.1: RAG enhancer (lazy initialization)
        self._rag_enhancer = None
    
    def _process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Build context for generation.
        
        Phase 7.1: Enhanced with RAG (Retrieval-Augmented Generation).
        """
        # Tool selection (if tool_policy available)
        if self.tool_policy:
            try:
                context.tool_name = self.tool_policy.choose_tool(context.features)
            except Exception:
                context.tool_name = None
        
        # Summarize if needed
        history = self.memory_service.summarize_if_needed(context.state.history)
        
        # Prune
        history = self.memory_service.prune(history, user_message=context.request.user_message)
        
        # Build base context
        context_text = self.memory_service.build_context(
            user_message=context.request.user_message,
            history=history,
            system_prompt=None,  # System prompt artık context'e dahil edilmiyor
        )
        
        # Phase 7.1: Enhance context with RAG if enabled
        try:
            # Lazy initialize RAG enhancer
            if self._rag_enhancer is None:
                cfg = getattr(self.memory_service, 'cfg', None)
                if cfg and cfg.memory.enable_rag:
                    from cognitive_management.v2.components.rag_enhancer import RAGEnhancer
                    self._rag_enhancer = RAGEnhancer(self.memory_service, cfg)
            
            # Apply RAG enhancement
            if self._rag_enhancer and self._rag_enhancer.enabled:
                context_text = self._rag_enhancer.enhance_context(
                    user_message=context.request.user_message,
                    existing_context=context_text,
                )
        except Exception as e:
            # Log warning but continue without RAG enhancement
            import logging
            logging.warning(f"RAG enhancement başarısız, devam ediliyor: {e}")
        
        # Add internal thought if exists (CoT reasoning)
        if context.selected_thought and hasattr(context.selected_thought, 'text') and context.selected_thought.text:
            context_text = f"{context_text}\n\n[INTERNAL SELECTED]\n{context.selected_thought.text}"
        
        # Add tool request if exists
        if context.tool_name:
            context_text = f"{context_text}\n\n[TOOL_REQUEST] {context.tool_name}"
        
        context.context_text = context_text
        return context


# =============================================================================
# Generation Handler
# =============================================================================

class GenerationHandler(BaseProcessingHandler):
    """
    Generation handler.
    SOLID: SRP - Sadece text generation yapar.
    """
    
    def __init__(self, backend):
        super().__init__("Generation")
        self.backend = backend
    
    def _process(self, context: ProcessingContext) -> ProcessingContext:
        """Generate text using backend"""
        policy = context.policy_output
        
        if not policy or not context.context_text:
            context.errors.append("Generation: Missing policy or context")
            return context
        
        # Generate
        try:
            draft = self.backend.generate(
                prompt=context.context_text,
                decoding_config=policy.decoding,
            )
            context.draft_text = (draft or "").strip()
        except Exception as e:
            context.errors.append(f"Generation error: {e}")
            context.draft_text = ""
        
        return context


# =============================================================================
# Critic Handler
# =============================================================================

class CriticHandler(BaseProcessingHandler):
    """
    Critic handler.
    SOLID: SRP - Sadece critic review yapar.
    """
    
    def __init__(self, critic):
        super().__init__("Critic")
        self.critic = critic
    
    def _process(self, context: ProcessingContext) -> ProcessingContext:
        """Review and revise text"""
        if not context.draft_text:
            return context
        
        # Review
        try:
            final_text, revised = self.critic.review(
                user_message=context.request.user_message,
                draft_text=context.draft_text,
                context=context.context_text,
            )
            context.final_text = final_text
            context.revised = revised
        except Exception:
            # Critic hatası ana akışı düşürmesin
            context.final_text = context.draft_text
            context.revised = False
        
        return context


# =============================================================================
# Memory Update Handler
# =============================================================================

class MemoryUpdateHandler(BaseProcessingHandler):
    """
    Memory update handler.
    SOLID: SRP - Sadece memory update yapar.
    """
    
    def __init__(self, memory_service):
        super().__init__("MemoryUpdate")
        self.memory_service = memory_service
    
    def _process(self, context: ProcessingContext) -> ProcessingContext:
        """Update memory with conversation turn"""
        final_text = context.final_text or context.draft_text or ""
        
        if final_text:
            # Add user turn
            self.memory_service.add_turn(
                history=context.state.history,
                role="user",
                content=context.request.user_message,
            )
            
            # Add assistant turn
            self.memory_service.add_turn(
                history=context.state.history,
                role="assistant",
                content=final_text,
            )
            
            # Inject session summary if needed (after adding assistant turn)
            try:
                self.memory_service.inject_session_summary_if_needed(context.state.history)
            except Exception:
                # Non-critical - don't fail the request if summary injection fails
                pass
            
            # Update state
            context.state.step += 1
            if context.policy_output:
                context.state.last_mode = context.policy_output.mode
            if context.features:
                context.state.last_entropy = float(context.features.get("entropy_est", 0.0))
        
        return context


__all__ = [
    "FeatureExtractionHandler",
    "PolicyRoutingHandler",
    "DeliberationHandler",
    "ContextBuildingHandler",
    "GenerationHandler",
    "CriticHandler",
    "MemoryUpdateHandler",
]

