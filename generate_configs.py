from pydantic import BaseModel
from typing import Literal


class Config(BaseModel):
    DEVICE: int = 0
    DATASET: Literal["shakespeare", "custom"]
    BASE: Literal["byte", "char", "sub", "word"]
    E2E: Literal["true", "false"]
    LANG: Literal["en", "ru", "fr", "hi"]
    LEARNING_RATE: float = 1e-4
    NUM_PREFIX: int = 4
    BATCH_SIZE: int = 2
    EPOCHS: int = 100
    OPENBLAS_NUM_THREADS: int = 3
    WANDB_API_KEY: str = "c8032f09362a503f2d512fad39cb26bb71eba5ac"
    WANDB_MODE: str = "offline"


ru_config = [
    {
        "DATASET": "custom",
        "BASE": "byte",
        "E2E": "false",
        "LANG": "ru",
    },
    {
        "DATASET": "custom",
        "BASE": "char",
        "E2E": "false",
        "LANG": "ru",
    },
    {
        "DATASET": "custom",
        "BASE": "sub",
        "E2E": "false",
        "LANG": "ru",
    },
    {
        "DATASET": "custom",
        "BASE": "word",
        "E2E": "false",
        "LANG": "ru",
    },
    {
        "DATASET": "custom",
        "BASE": "byte",
        "E2E": "true",
        "LANG": "ru",
    },
    {
        "DATASET": "custom",
        "BASE": "char",
        "E2E": "true",
        "LANG": "ru",
    },
    {
        "DATASET": "custom",
        "BASE": "sub",
        "E2E": "true",
        "LANG": "ru",
    },
]

en_config = [
    {
        "DATASET": "custom",
        "BASE": "byte",
        "E2E": "false",
        "LANG": "en",
    },
    {
        "DATASET": "custom",
        "BASE": "char",
        "E2E": "false",
        "LANG": "en",
    },
    {
        "DATASET": "custom",
        "BASE": "sub",
        "E2E": "false",
        "LANG": "en",
    },
    {
        "DATASET": "custom",
        "BASE": "word",
        "E2E": "false",
        "LANG": "en",
    },
    {
        "DATASET": "custom",
        "BASE": "byte",
        "E2E": "true",
        "LANG": "en",
    },
    {
        "DATASET": "custom",
        "BASE": "char",
        "E2E": "true",
        "LANG": "en",
    },
    {
        "DATASET": "custom",
        "BASE": "sub",
        "E2E": "true",
        "LANG": "en",
    },
]

fr_config = [
    {
        "DATASET": "custom",
        "BASE": "byte",
        "E2E": "false",
        "LANG": "fr",
    },
    {
        "DATASET": "custom",
        "BASE": "char",
        "E2E": "false",
        "LANG": "fr",
    },
    {
        "DATASET": "custom",
        "BASE": "sub",
        "E2E": "false",
        "LANG": "fr",
    },
    {
        "DATASET": "custom",
        "BASE": "word",
        "E2E": "false",
        "LANG": "fr",
    },
    {
        "DATASET": "custom",
        "BASE": "byte",
        "E2E": "true",
        "LANG": "fr",
    },
    {
        "DATASET": "custom",
        "BASE": "char",
        "E2E": "true",
        "LANG": "fr",
    },
    {
        "DATASET": "custom",
        "BASE": "sub",
        "E2E": "true",
        "LANG": "fr",
    },
]

shakespeare_config = [
    {
        "DATASET": "shakespeare",
        "BASE": "byte",
        "E2E": "false",
        "LANG": "en",
    },
    {
        "DATASET": "shakespeare",
        "BASE": "char",
        "E2E": "false",
        "LANG": "en",
    },
    {
        "DATASET": "shakespeare",
        "BASE": "sub",
        "E2E": "false",
        "LANG": "en",
    },
    {
        "DATASET": "shakespeare",
        "BASE": "word",
        "E2E": "false",
        "LANG": "en",
    },
    {
        "DATASET": "shakespeare",
        "BASE": "byte",
        "E2E": "true",
        "LANG": "en",
    },
    {
        "DATASET": "shakespeare",
        "BASE": "char",
        "E2E": "true",
        "LANG": "en",
    },
    {
        "DATASET": "shakespeare",
        "BASE": "sub",
        "E2E": "true",
        "LANG": "en",
    },
]


# ru_config = [
#     {
#         "DATASET": "custom",
#         "BASE": "byte",
#         "E2E": "true",
#         "LANG": "ru",
#         "NUM_PREFIX": 1
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "char",
#         "E2E": "true",
#         "LANG": "ru",
#         "NUM_PREFIX": 1
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "sub",
#         "E2E": "true",
#         "LANG": "ru",
#         "NUM_PREFIX": 1
#     },
# ]

# en_config = [
#     {
#         "DATASET": "custom",
#         "BASE": "byte",
#         "E2E": "true",
#         "LANG": "en",
#         "NUM_PREFIX": 1
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "char",
#         "E2E": "true",
#         "LANG": "en",
#         "NUM_PREFIX": 1
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "sub",
#         "E2E": "true",
#         "LANG": "en",
#         "NUM_PREFIX": 1
#     },
# ]

# fr_config = [
#     {
#         "DATASET": "custom",
#         "BASE": "byte",
#         "E2E": "true",
#         "LANG": "fr",
#         "NUM_PREFIX": 1
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "char",
#         "E2E": "true",
#         "LANG": "fr",
#         "NUM_PREFIX": 1
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "sub",
#         "E2E": "true",
#         "LANG": "fr",
#         "NUM_PREFIX": 1
#     },
# ]

# shakespeare_config = [
#     {
#         "DATASET": "shakespeare",
#         "BASE": "byte",
#         "E2E": "true",
#         "LANG": "en",
#         "NUM_PREFIX": 1
#     },
#     {
#         "DATASET": "shakespeare",
#         "BASE": "char",
#         "E2E": "true",
#         "LANG": "en",
#         "NUM_PREFIX": 1
#     },
#     {
#         "DATASET": "shakespeare",
#         "BASE": "sub",
#         "E2E": "true",
#         "LANG": "en",
#         "NUM_PREFIX": 1
#     },
    
# ]

for config in [*ru_config, *en_config, *fr_config, *shakespeare_config]:
    cfg = Config(**config)
    with open(
        f"configs/{cfg.DATASET}_{cfg.LANG}_{cfg.BASE}_{'e2e' if cfg.E2E== 'true' else 'no-e2e'}_{cfg.LEARNING_RATE}_{cfg.NUM_PREFIX}_{cfg.BATCH_SIZE}.env",
        "w",
    ) as f:
        f.write(
            f"WANDB_MODE={cfg.WANDB_MODE}\nDEVICE={cfg.DEVICE}\nDATASET={cfg.DATASET}\nBASE={cfg.BASE}\nE2E={cfg.E2E}\nLANG={cfg.LANG}\nLEARNING_RATE={cfg.LEARNING_RATE}\nNUM_PREFIX={cfg.NUM_PREFIX}\nBATCH_SIZE={cfg.BATCH_SIZE}\nEPOCHS={cfg.EPOCHS}\nOPENBLAS_NUM_THREADS={cfg.OPENBLAS_NUM_THREADS}\nWANDB_API_KEY={cfg.WANDB_API_KEY}\n"
        )
    print(
        f"configs/{cfg.DATASET}_{cfg.LANG}_{cfg.BASE}_{'e2e' if cfg.E2E== 'true' else 'no-e2e'}_{cfg.LEARNING_RATE}_{cfg.NUM_PREFIX}_{cfg.BATCH_SIZE}.env"
    )
