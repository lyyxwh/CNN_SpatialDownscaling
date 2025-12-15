# configs/common.py (片段示例)
common_config = {
    "training": {
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 100,
        "early_stopping_patience": 10,
        "time_window_days": 30,
        "device": "cuda"
    },

    # Landcover: CLCD 本身已编码，下面字典仅用于可读性/分析，不用于再编码
    # 如果 CLCD 原始编码不是 1..N，请在 dataset 中按需要映射到连续的类别索引
    "landcover_classes": {
        1: "cropland",
        2: "forest",
        3: "grassland",
        4: "urban",
        5: "water",
        # 若 CLCD 有更多类别，在此补全（仅作 label 用）
    },
    "landcover_nodata_value": 0,    # 若 CLCD 使用 0 表示nodata，写明；否则在 dataset 中映射

    # SMAP 土壤湿度标签编码（已修正）：0 表示缺失，1-5 为不同湿度段
    "smap_encoding": {
        "nodata": 0,
        "0-20%": 1,
        "20-40%": 2,
        "40-60%": 3,
        "60-80%": 4,
        "80-100%": 5
    },
    "smap_nodata_value": 0,   # 明确缺测对应的数值

    # Embedding 配置（建议）
    "embeddings": {
        "landcover_embedding_dim": 16,   # CLCD embedding 维度（可调整）
        "smap_embedding_dim": 8          # SMAP embedding 维度
    },

    # 标准化参数占位（训练时计算并回写到此处或单独保存）
    "normalization": {
        "t2m": {"mean": None, "std": None},
        "ssrd": {"mean": None, "std": None},
        "strd": {"mean": None, "std": None},
        "vpd": {"mean": None, "std": None},
        "d2m": {"mean": None, "std": None},
        "rh": {"mean": None, "std": None},
        "dem": {"mean": None, "std": None},
        "slope": {"mean": None, "std": None},
        "aspect_sin": {"mean": None, "std": None},
        "aspect_cos": {"mean": None, "std": None},
        "ndvi": {"mean": None, "std": None}
    }
}
