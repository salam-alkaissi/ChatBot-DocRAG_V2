# src/model_config.py
MODEL_PATHS = {
    "domain_classifier": {
        "path": "models/domain_classifier",
        "version": "v2.1"
    },
    "pdf_extractor": {
        "path": "models/layoutlm",
        "version": "v1.4"
    }
}

DEVICE_CONFIG = {
    "use_gpu": True,
    "max_memory": 0.8  # 80% of available GPU memory
}