# -*- coding: utf-8 -*-
"""
================================================================================
CEVAHIR-AI PROJESİ
================================================================================

Dosya: tree_of_thoughts.py
Modül: cognitive_management/v2/components
Görev: Tree of Thoughts (ToT) - Tree of Thoughts reasoning pattern for complex
       problem solving. Phase 4: Advanced Reasoning Patterns. Thought generation,
       evaluation, expansion ve selection işlemlerini yapar. Akademik referans:
       Yao et al. (2023).

MİMARİ:
- SOLID Prensipleri: Single Responsibility (ToT reasoning),
                     Dependency Inversion (interface'lere bağımlı)
- Design Patterns: Tree Pattern (tree of thoughts)
- Endüstri Standartları: Advanced reasoning patterns

KULLANIM:
- Complex problem solving için
- Thought generation için
- Thought evaluation için

BAĞIMLILIKLAR:
- ModelAPI: Model interface
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
from typing import List, Optional, Protocol, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from cognitive_management.cognitive_types import DecodingConfig
from cognitive_management.config import CognitiveManagerConfig
from cognitive_management.exceptions import ValidationError


class ModelAPI(Protocol):
    """Model API interface for Tree of Thoughts."""
    def generate(self, prompt: str, decoding_cfg: DecodingConfig) -> str: ...
    def score(self, prompt: str, candidate: str) -> float: ...


class ThoughtState(Enum):
    """State of a thought node in the tree."""
    EXPANDED = "expanded"  # Has been expanded (children generated)
    LEAF = "leaf"  # Leaf node (no children)
    PRUNED = "pruned"  # Pruned from tree (low score)


@dataclass
class ThoughtNode:
    """
    Node in the Tree of Thoughts.
    
    Represents a single reasoning step or thought.
    
    Attributes:
        thought: The thought/reasoning step text
        parent: Parent node (None for root)
        children: Child nodes (thought expansions)
        score: Evaluation score (0.0-1.0, higher is better)
        depth: Depth in tree (0 for root)
        state: Current state of the node
    """
    thought: str
    parent: Optional[ThoughtNode] = None
    children: List[ThoughtNode] = field(default_factory=list)
    score: float = 0.0
    depth: int = 0
    state: ThoughtState = ThoughtState.LEAF
    
    def add_child(self, child: ThoughtNode) -> None:
        """Add a child node to this node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        if self.state == ThoughtState.LEAF:
            self.state = ThoughtState.EXPANDED
    
    def get_path_to_root(self) -> List[ThoughtNode]:
        """Get path from this node to root."""
        path = []
        current = self
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0 and self.state != ThoughtState.PRUNED


class TreeOfThoughts:
    """
    Tree of Thoughts reasoning engine.
    
    Builds a tree of reasoning paths, expands promising nodes,
    and selects the best path(s) through the tree.
    
    Academic Reference:
    - Yao et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
    """
    
    def __init__(
        self,
        cfg: CognitiveManagerConfig,
        model_api: ModelAPI,
        max_depth: int = 3,
        branching_factor: int = 3,
        top_k: int = 5
    ):
        """
        Initialize Tree of Thoughts engine.
        
        Args:
            cfg: Cognitive manager configuration
            model_api: Model API for generation
            max_depth: Maximum tree depth
            branching_factor: Number of children to generate per node
            top_k: Number of top paths to keep
        """
        if not isinstance(cfg, CognitiveManagerConfig):
            raise ValidationError("cfg tipi geçersiz (CognitiveManagerConfig bekleniyor).")
        
        self.cfg = cfg
        self.mm = model_api
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.top_k = top_k
    
    def solve(
        self,
        problem: str,
        system_prompt: Optional[str] = None,
        decoding_config: Optional[DecodingConfig] = None
    ) -> List[Tuple[List[str], float]]:
        """
        Solve a problem using Tree of Thoughts reasoning.
        
        Args:
            problem: Problem statement or question
            system_prompt: Optional system prompt
            decoding_config: Optional decoding configuration
            
        Returns:
            List of (path, score) tuples, where path is list of thoughts
            and score is the path evaluation score. Sorted by score (best first).
        """
        if not problem or not problem.strip():
            raise ValidationError("Problem boş olamaz.")
        
        system_prompt = system_prompt or self.cfg.default_system_prompt
        decoding_config = decoding_config or self.cfg.default_decoding
        
        # Create root node
        root = ThoughtNode(
            thought="Initial problem analysis",
            depth=0,
            score=0.5,  # Neutral initial score
        )
        
        # Build the tree
        tree = self._build_tree(root, problem, system_prompt, decoding_config)
        
        # Find best paths
        best_paths = self._find_best_paths(tree, top_k=self.top_k)
        
        # Convert to output format
        results = []
        for node, path_score in best_paths:
            path = [n.thought for n in node.get_path_to_root()]
            results.append((path, path_score))
        
        return results
    
    def _build_tree(
        self,
        root: ThoughtNode,
        problem: str,
        system_prompt: str,
        decoding_config: DecodingConfig
    ) -> ThoughtNode:
        """
        Build the tree of thoughts by expanding nodes.
        
        Uses breadth-first expansion, expanding nodes at each level
        before moving to the next level.
        
        Args:
            root: Root node of the tree
            problem: Problem statement
            system_prompt: System prompt
            decoding_config: Decoding configuration
            
        Returns:
            Root node of the built tree
        """
        # Queue for breadth-first expansion
        current_level = [root]
        
        for depth in range(self.max_depth):
            if not current_level:
                break
            
            next_level = []
            
            # Expand all nodes at current depth
            for node in current_level:
                if node.state == ThoughtState.PRUNED:
                    continue
                
                # Generate children (thought expansions)
                children = self._expand_node(
                    node, problem, system_prompt, decoding_config
                )
                
                # Evaluate children
                evaluated_children = []
                for child in children:
                    score = self._evaluate_thought(
                        child, problem, system_prompt, decoding_config
                    )
                    child.score = score
                    evaluated_children.append(child)
                
                # Phase 8: Enhanced selection using selectors utilities
                # Convert ThoughtNode to ThoughtCandidate for selector functions
                from cognitive_management.cognitive_types import ThoughtCandidate
                thought_candidates = [
                    ThoughtCandidate(text=child.thought, score=child.score)
                    for child in evaluated_children
                ]
                
                # Use select_topk for better top-k selection
                from ..utils.selectors import select_topk
                top_candidates = select_topk(thought_candidates, k=self.branching_factor)
                
                # Map back to ThoughtNode and keep top branching_factor
                top_children = []
                candidate_dict = {tc.text: tc.score for tc in top_candidates}
                for child in evaluated_children:
                    if child.thought in candidate_dict and candidate_dict[child.thought] == child.score:
                        top_children.append(child)
                        if len(top_children) >= self.branching_factor:
                            break
                
                # Fallback: If selection failed, use simple sort
                if len(top_children) < self.branching_factor:
                    evaluated_children.sort(key=lambda n: n.score, reverse=True)
                    top_children = evaluated_children[:self.branching_factor]
                
                # Add top children to node
                for child in top_children:
                    node.add_child(child)
                    next_level.append(child)
                
                # Prune low-scoring children
                for child in evaluated_children[self.branching_factor:]:
                    child.state = ThoughtState.PRUNED
            
            # Move to next level
            current_level = next_level
        
        return root
    
    def _expand_node(
        self,
        node: ThoughtNode,
        problem: str,
        system_prompt: str,
        decoding_config: DecodingConfig
    ) -> List[ThoughtNode]:
        """
        Expand a node by generating child thoughts.
        
        Generates multiple alternative reasoning steps from the current thought.
        
        Args:
            node: Node to expand
            problem: Original problem
            system_prompt: System prompt
            decoding_config: Decoding configuration
            
        Returns:
            List of child ThoughtNodes
        """
        # Build context from path to root
        path_to_root = node.get_path_to_root()
        context = "\n".join([f"Step {i+1}: {n.thought}" for i, n in enumerate(path_to_root)])
        
        # Build expansion prompt
        prompt = self._build_expansion_prompt(
            problem, context, node.thought, system_prompt
        )
        
        # Generate multiple thought continuations
        children = []
        num_children = self.branching_factor + 1  # Generate one extra for pruning
        
        for i in range(num_children):
            try:
                # Generate thought continuation
                thought_text = self.mm.generate(prompt, decoding_config)
                thought_text = (thought_text or "").strip()
                
                if thought_text:
                    child = ThoughtNode(thought=thought_text, parent=node)
                    children.append(child)
            except Exception:
                # Generation failed - skip this child
                continue
        
        return children
    
    def _build_expansion_prompt(
        self,
        problem: str,
        context: str,
        current_thought: str,
        system_prompt: str
    ) -> str:
        """
        Build prompt for thought expansion.
        
        Args:
            problem: Original problem
            context: Current reasoning path
            current_thought: Current thought to expand
            system_prompt: System prompt
            
        Returns:
            Expansion prompt
        """
        return f"""[SYSTEM]
{system_prompt}

[PROBLEM]
{problem}

[REASONING PATH SO FAR]
{context}

[CURRENT THOUGHT]
{current_thought}

[INSTRUCTION]
The above reasoning path has led to the current thought. Now, think of the next logical step or alternative approach to continue solving this problem.

Generate a concise next reasoning step. Consider:
1. What is the next logical step?
2. Are there alternative approaches?
3. What information or analysis is needed next?

Next step:"""
    
    def _evaluate_thought(
        self,
        node: ThoughtNode,
        problem: str,
        system_prompt: str,
        decoding_config: DecodingConfig
    ) -> float:
        """
        Evaluate a thought node.
        
        Scores the quality and relevance of a thought in the context
        of the problem and reasoning path.
        
        Args:
            node: Thought node to evaluate
            problem: Original problem
            system_prompt: System prompt
            decoding_config: Decoding configuration
            
        Returns:
            Evaluation score (0.0-1.0, higher is better)
        """
        # Build evaluation context
        path_to_root = node.get_path_to_root()
        context = "\n".join([f"Step {i+1}: {n.thought}" for i, n in enumerate(path_to_root)])
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            problem, context, node.thought, system_prompt
        )
        
        try:
            # Use model scoring if available
            score = self.mm.score(prompt, node.thought)
            score = float(score)
            
            # Normalize to 0.0-1.0 range
            if score < 0:
                score = 0.0
            elif score > 1.0:
                score = score / 10.0  # Assume scale of 10
                score = min(1.0, max(0.0, score))
            
            return score
            
        except Exception:
            # Scoring failed - use heuristic
            return self._heuristic_evaluate(node, problem)
    
    def _build_evaluation_prompt(
        self,
        problem: str,
        context: str,
        thought: str,
        system_prompt: str
    ) -> str:
        """
        Build prompt for thought evaluation.
        
        Args:
            problem: Original problem
            context: Reasoning path context
            thought: Thought to evaluate
            system_prompt: System prompt
            
        Returns:
            Evaluation prompt
        """
        return f"""[SYSTEM]
{system_prompt}

[PROBLEM]
{problem}

[REASONING PATH]
{context}

[CURRENT THOUGHT TO EVALUATE]
{thought}

[INSTRUCTION]
Evaluate how well this thought contributes to solving the problem. Consider:
1. Is it relevant to the problem?
2. Does it make logical sense in the context?
3. Does it advance the solution?
4. Is it clear and well-reasoned?

Rate from 0.0 to 1.0 (1.0 = excellent contribution):"""
    
    def _heuristic_evaluate(self, node: ThoughtNode, problem: str) -> float:
        """
        Heuristic evaluation of a thought (fallback).
        
        Args:
            node: Thought node
            problem: Original problem
            
        Returns:
            Heuristic score (0.0-1.0)
        """
        thought = node.thought.lower()
        problem_lower = problem.lower()
        
        score = 0.3  # Base score
        
        # Relevance: Keyword overlap with problem
        problem_words = set(problem_lower.split())
        thought_words = set(thought.split())
        overlap = len(problem_words & thought_words)
        if problem_words:
            relevance = overlap / len(problem_words)
            score += relevance * 0.3
        
        # Length: Reasonable length is better
        length = len(node.thought)
        if 50 <= length <= 300:
            score += 0.2
        elif 20 <= length < 50 or 300 < length <= 500:
            score += 0.1
        
        # Structure: Step indicators
        step_indicators = ["adım", "step", "sonraki", "next", "alternatif", "alternative"]
        if any(indicator in thought for indicator in step_indicators):
            score += 0.2
        
        return min(1.0, score)
    
    def _find_best_paths(
        self,
        root: ThoughtNode,
        top_k: int = 5
    ) -> List[Tuple[ThoughtNode, float]]:
        """
        Find the best paths through the tree.
        
        Evaluates all leaf nodes (complete paths) and returns
        the top-k paths with highest scores.
        
        Args:
            root: Root node of tree
            top_k: Number of top paths to return
            
        Returns:
            List of (leaf_node, path_score) tuples, sorted by score
        """
        # Find all leaf nodes (complete paths)
        leaves = self._find_leaves(root)
        
        # Evaluate each path
        path_scores = []
        for leaf in leaves:
            path_score = self._evaluate_path(leaf)
            path_scores.append((leaf, path_score))
        
        # Sort by score and return top-k
        path_scores.sort(key=lambda x: x[1], reverse=True)
        return path_scores[:top_k]
    
    def _find_leaves(self, root: ThoughtNode) -> List[ThoughtNode]:
        """
        Find all leaf nodes in the tree.
        
        Args:
            root: Root node
            
        Returns:
            List of leaf nodes
        """
        leaves = []
        
        def dfs(node: ThoughtNode):
            if node.is_leaf():
                leaves.append(node)
            else:
                for child in node.children:
                    if child.state != ThoughtState.PRUNED:
                        dfs(child)
        
        dfs(root)
        return leaves
    
    def _evaluate_path(self, leaf: ThoughtNode) -> float:
        """
        Evaluate a complete path from root to leaf.
        
        Combines scores of all nodes in the path.
        
        Args:
            leaf: Leaf node (end of path)
            
        Returns:
            Path evaluation score
        """
        path = leaf.get_path_to_root()
        
        if not path:
            return 0.0
        
        # Average score of all nodes in path
        avg_score = sum(node.score for node in path) / len(path)
        
        # Bonus for longer paths (more reasoning steps)
        length_bonus = min(0.1, len(path) * 0.02)
        
        return min(1.0, avg_score + length_bonus)


__all__ = [
    "TreeOfThoughts",
    "ThoughtNode",
    "ThoughtState",
    "ModelAPI",
]

