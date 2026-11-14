"""Roof data: what each facade exports, which old paths are deprecated, who owns which responsibility.

Pure data; importing this module imports nothing else. ``EXPORTS`` maps
``facade -> {public name: "legacy.module:attribute"}``. A facade never copies or wraps: the public
name is the legacy object itself. Only the live member of each v2/v3 pair is exported, under an
unversioned name; duplicated names get one roof name each (``SystemConfig`` and ``FacadeConfig``
for the two ``CevahirConfig`` classes).
"""
from dataclasses import dataclass

EXPORTS = {
    "neural": {
        "CevahirNeuralNetwork": "src.neural_network:CevahirNeuralNetwork",
        "resolve_ffn_dim": "src.neural_network_module.architecture_contracts:resolve_ffn_dim",
        "TransformerEncoderLayer": "src.neural_network_module.ortak_katman_module.transformer_encoder_layer:TransformerEncoderLayer",
        "MultiHeadAttention":
            "src.neural_network_module.ortak_katman_module.attention_manager_module.multi_head_attention:MultiHeadAttention",
        "FeedForwardNetwork": "src.neural_network_module.ortak_katman_module.feed_forward_network:FeedForwardNetwork",
        "MixtureOfExperts": "src.neural_network_module.ortak_katman_module.mixture_of_experts:MixtureOfExperts",
        "RMSNorm": "src.neural_network_module.ortak_katman_module.rms_norm:RMSNorm",
        "LanguageEmbedding": "src.neural_network_module.dil_katmani_module.language_embedding:LanguageEmbedding",
        "PositionalEncoding": "src.neural_network_module.dil_katmani_module.positional_encoding:PositionalEncoding",
    },
    "model": {
        "ModelManager": "model_management.model_manager:ModelManager",
        "ModelInitializer": "model_management.model_initializer:ModelInitializer",
        "ModelSaver": "model_management.model_saver:ModelSaver",
        "ModelLoader": "model_management.model_loader:ModelLoader",
    },
    "model.config": {
        "ModelArchConfig": "model_management.config_schema:ModelArchConfig",
        "TrainingConfig": "model_management.config_schema:TrainingConfig",
        "CheckpointConfig": "model_management.config_schema:CheckpointConfig",
        "DistributedConfig": "model_management.config_schema:DistributedConfig",
        "QuantConfig": "model_management.config_schema:QuantConfig",
        "SystemConfig": "model_management.config_schema:CevahirConfig",
        "normalize_model_config": "model_management.config_schema:normalize_model_config",
        "tiny_model_config": "model_management.config_schema:tiny_model_config",
        "CONFIG_VERSION": "model_management.config_schema:CONFIG_VERSION",
        "ARCHITECTURE_VERSION": "model_management.config_schema:ARCHITECTURE_VERSION",
    },
    "model.checkpoint": {
        "unpack_checkpoint": "model_management.checkpoint_contract:unpack_checkpoint",
        "model_config": "model_management.checkpoint_contract:model_config",
        "validate_model_identity": "model_management.checkpoint_contract:validate_model_identity",
        "validate_state_shapes": "model_management.checkpoint_contract:validate_state_shapes",
        "validate_tokenizer_identity": "model_management.checkpoint_contract:validate_tokenizer_identity",
        "validate_scheduler_state": "model_management.checkpoint_contract:validate_scheduler_state",
    },
    "tokenizer": {
        "LegacyBPETokenizer": "tokenizer_management.core.tokenizer_core:TokenizerCore",
        "ByteLevelBPE": "tokenizer_management.byte_bpe:ByteLevelBPE",
    },
    "data": {
        "DataCache": "training_system.data_cache:DataCache",
        "StrictCache": "training_system.v3.data.cache_v3:DataCacheV3",
        "split_training_records": "training_system.data_split:split_training_records",
        "record_format": "training_system.record_format",
        "DataLoaderManager": "data_loader_management.data_loader_manager:DataLoaderManager",
    },
    "training": {
        "TrainingService": "training_system.v3.core.training_service_v3:TrainingServiceV3",
        "TrainingEngine": "training_management.v2.core.training_manager:TrainingManager",
        "CheckpointManager": "training_management.v2.utils.checkpoint_manager:CheckpointManager",
        "TrainingScheduler": "training_management.v2.utils.training_scheduler:TrainingScheduler",
        "validate_training_backend": "training_management.contracts:validate_training_backend",
        "resolve_precision": "training_management.contracts:resolve_precision",
    },
    "cognition": {
        "CognitiveManager": "cognitive_management.cognitive_manager:CognitiveManager",
        "CognitiveManagerConfig": "cognitive_management.config:CognitiveManagerConfig",
        "types": "cognitive_management.cognitive_types",
    },
    "serving": {
        "Cevahir": "model.cevahir:Cevahir",
        "FacadeConfig": "model.cevahir:CevahirConfig",
        "create_app": "api.app_factory:create_app",
        "ChattingManager": "chatting_management.chatting_manager:ChattingManager",
    },
}

# Never exported by the roof: the fallback or dead member of a duplicated pair.
NOT_EXPORTED = (
    "training_system.v2.core.training_service", "training_management.v3", "training_system.v2.core.data_preparator",
    "model_management.chat_pipeline", "api.app", "api.routes.chat_routes_v2", "training_system.config_validator",
)

# old module path -> (replacement, deprecated in, reason). Data only: nothing warns until a module moves.
DEPRECATED = {
    "training_system.v2.core.training_service": ("cevahir_ai.training.TrainingService", "0.2.0",
                                                 "fallback service; train.py selects the v3 service"),
    "training_system.v2.core.data_preparator": ("cevahir_ai.data", "0.2.0", "not on the live training path"),
    "training_management.v3": (None, "0.2.0", "imported by train.py only to set a log flag; never called"),
    "model_management.chat_pipeline": ("cevahir_ai.serving.Cevahir", "0.2.0", "legacy REPL; logs to a file at import"),
    "training_system.debug_all_cache": ("python -m cevahir_ai.keys explain <cache file>", "0.2.0",
                                        "computes an old md5 that no longer matches real cache keys"),
}


@dataclass(frozen=True)
class Responsibility:
    id: str
    title: str
    owner: str                 # the roof module that owns it
    implementations: tuple     # legacy modules that implement it today
    status: str                # "owned" (roof code exists) | "facade" (roof name over legacy code) | "planned"


RESPONSIBILITIES = (
    Responsibility("R-IDENTITY", "Artifact identity and keys", "cevahir_ai.keys",
                   ("training_system.cache_identity", "model_management.checkpoint_contract",
                    "model_management.model_saver", "tokenizer_management.byte_bpe",
                    "training_system.data_cache", "training_system.v3.data.cache_v3"), "owned"),
    Responsibility("R-COMPAT", "Compatibility decisions at load", "cevahir_ai.keys.gate",
                   ("model_management.checkpoint_contract",), "owned"),
    Responsibility("R-CKPT-IO", "Checkpoint read and write", "cevahir_ai.model.checkpoint",
                   ("model_management.model_saver", "model_management.model_loader",
                    "training_management.v2.utils.checkpoint_manager"), "owned"),
    Responsibility("R-EVIDENCE", "Frozen-study dependency pin", "cevahir_ai.evidence", (), "owned"),
    Responsibility("R-NEURAL", "Neural core", "cevahir_ai.neural", ("src.neural_network",), "facade"),
    Responsibility("R-MODEL-LIFECYCLE", "Model lifecycle", "cevahir_ai.model",
                   ("model_management.model_manager", "model_management.model_initializer"), "facade"),
    Responsibility("R-TOKENIZER", "Tokenizers", "cevahir_ai.tokenizer",
                   ("tokenizer_management.core.tokenizer_core", "tokenizer_management.byte_bpe"), "facade"),
    Responsibility("R-CACHE", "Training data cache", "cevahir_ai.data",
                   ("training_system.data_cache", "training_system.v3.data.cache_v3"), "facade"),
    Responsibility("R-DATA-LOAD", "Batching and data loading", "cevahir_ai.data",
                   ("data_loader_management.data_loader_manager",), "facade"),
    Responsibility("R-TRAIN-ORCH", "Training orchestration", "cevahir_ai.training",
                   ("training_system.v3.core.training_service_v3",), "facade"),
    Responsibility("R-TRAIN-ENGINE", "Training engine", "cevahir_ai.training",
                   ("training_management.v2.core.training_manager",), "facade"),
    Responsibility("R-COGNITION", "Cognition", "cevahir_ai.cognition", ("cognitive_management.cognitive_manager",), "facade"),
    Responsibility("R-SERVE", "Serving", "cevahir_ai.serving",
                   ("model.cevahir", "api.app_factory", "chatting_management.chatting_manager"), "facade"),
    Responsibility("R-AUDIT", "Registry, audit log and lineage", "cevahir_ai.store", (), "owned"),
    Responsibility("R-POLICY", "Policy and approvals", "cevahir_ai.gov", ("training_management.contracts",), "planned"),
    Responsibility("R-OPERATOR", "Operator console", "cevahir_ai.console", (), "owned"),
)
