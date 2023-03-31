from pydantic import BaseModel
from typing import Literal


class ValConfig(BaseModel):
    LOAD_CKPT: str
    DEVICE: int = 0
    DATASET: Literal["shakespeare", "custom"]
    LANG: Literal["en", "ru", "fr", "hi"]
    BASE: Literal["byte", "char", "sub", "word"]
    E2E: Literal["true", "false"]
    USE_LOGGER: Literal["true", "false"] = "false"
    LEARNING_RATE: float = 1e-4
    NUM_PREFIX: int = 4
    BATCH_SIZE: int = 2
    OPENBLAS_NUM_THREADS: int = 3

# en_config = [
#     {
#         "DATASET": "custom",
#         "BASE": "byte",
#         "E2E": "false",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/kxc5tfyj/checkpoints/epoch=62-step=979524.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "char",
#         "E2E": "false",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/tshrru6g/checkpoints/epoch=71-step=1119456.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "word",
#         "E2E": "false",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/6ym5d28h/checkpoints/epoch=98-step=275121.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "byte",
#         "E2E": "true",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/fwp94zr0/checkpoints/epoch=98-step=275121.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "char",
#         "E2E": "true",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/lkmq8z0e/checkpoints/epoch=98-step=275121.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "sub",
#         "E2E": "true",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/zarb3lic/checkpoints/epoch=80-step=225099.ckpt",
#     },
# ]

# fr_config = [
#     {
#         "DATASET": "custom",
#         "BASE": "char",
#         "E2E": "false",
#         "LANG": "fr",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/n76bmh8w/checkpoints/epoch=59-step=1030800.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "sub",
#         "E2E": "false",
#         "LANG": "fr",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/czuqq74c/checkpoints/epoch=32-step=566940.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "word",
#         "E2E": "false",
#         "LANG": "fr",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/pbifulwi/checkpoints/epoch=98-step=317493.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "byte",
#         "E2E": "true",
#         "LANG": "fr",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/neslbjdm/checkpoints/epoch=74-step=240525.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "char",
#         "E2E": "true",
#         "LANG": "fr",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/phfttptc/checkpoints/epoch=98-step=317493.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "sub",
#         "E2E": "true",
#         "LANG": "fr",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/nk70pi95/checkpoints/epoch=53-step=173178.ckpt",
#     },
# ]

# ru_config = [
#     {
#         "DATASET": "custom",
#         "BASE": "byte",
#         "E2E": "false",
#         "LANG": "ru",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/i1q3oiwq/checkpoints/epoch=53-step=862866.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "char",
#         "E2E": "false",
#         "LANG": "ru",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/6rych95j/checkpoints/epoch=62-step=1006677.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "sub",
#         "E2E": "false",
#         "LANG": "ru",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/baf4jwnm/checkpoints/epoch=32-step=527307.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "word",
#         "E2E": "false",
#         "LANG": "ru",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/znw2h2on/checkpoints/epoch=98-step=241956.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "byte",
#         "E2E": "true",
#         "LANG": "ru",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/at2fy9mr/checkpoints/epoch=98-step=241956.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "char",
#         "E2E": "true",
#         "LANG": "ru",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/3xum2c7j/checkpoints/epoch=98-step=241956.ckpt",
#     },
#     {
#         "DATASET": "custom",
#         "BASE": "sub",
#         "E2E": "true",
#         "LANG": "ru",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/afkj9f4n/checkpoints/epoch=83-step=205296.ckpt",
#     },
# ]

# shakespeare_config = [
#     {
#         "DATASET": "shakespeare",
#         "BASE": "byte",
#         "E2E": "false",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/t45gj9nx/checkpoints/epoch=98-step=237600.ckpt",
#     },
#     {
#         "DATASET": "shakespeare",
#         "BASE": "char",
#         "E2E": "false",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/3uatax1k/checkpoints/epoch=98-step=237600.ckpt",
#     },
#     {
#         "DATASET": "shakespeare",
#         "BASE": "sub",
#         "E2E": "false",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/7y6fn93z/checkpoints/epoch=98-step=237600.ckpt",
#     },
#     {
#         "DATASET": "shakespeare",
#         "BASE": "word",
#         "E2E": "false",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/alkgjqf6/checkpoints/epoch=98-step=51876.ckpt",
#     },
#     {
#         "DATASET": "shakespeare",
#         "BASE": "byte",
#         "E2E": "true",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/4xafmof9/checkpoints/epoch=98-step=51876.ckpt",
#     },
#     {
#         "DATASET": "shakespeare",
#         "BASE": "char",
#         "E2E": "true",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/u107s5sz/checkpoints/epoch=98-step=51876.ckpt",
#     },
#     {
#         "DATASET": "shakespeare",
#         "BASE": "sub",
#         "E2E": "true",
#         "LANG": "en",
#         "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/due0lggq/checkpoints/epoch=98-step=51876.ckpt",
#     },
# ]

configs = [
    {
        "DATASET": "custom",
        "BASE": "byte",
        "E2E": "false",
        "LANG": "ru",
        "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/2ravqjw8/checkpoints/epoch=98-step=1581921.ckpt",
    },
    {
        "DATASET": "custom",
        "BASE": "sub",
        "E2E": "true",
        "LANG": "fr",
        "LOAD_CKPT": "/scratch1/sghaneka/etok/etok/kmxdnftd/checkpoints/epoch=98-step=317493.ckpt",
    },
]

for config in [*configs]:
    cfg = ValConfig(**config)
    with open(
        f"configs/val_{cfg.DATASET}_{cfg.LANG}_{cfg.BASE}_{'e2e' if cfg.E2E== 'true' else 'no-e2e'}_{cfg.LEARNING_RATE}_{cfg.NUM_PREFIX}_{cfg.BATCH_SIZE}.env",
        "w",
    ) as f:
        f.write(
            f"DEVICE={cfg.DEVICE}\nDATASET={cfg.DATASET}\nLOAD_CKPT={cfg.LOAD_CKPT}\nLANG={cfg.LANG}\nOPENBLAS_NUM_THREADS={cfg.OPENBLAS_NUM_THREADS}\nUSE_LOGGER={cfg.USE_LOGGER}"
        )
    print(
        f"configs/val_{cfg.DATASET}_{cfg.LANG}_{cfg.BASE}_{'e2e' if cfg.E2E== 'true' else 'no-e2e'}_{cfg.LEARNING_RATE}_{cfg.NUM_PREFIX}_{cfg.BATCH_SIZE}.env"
    )
