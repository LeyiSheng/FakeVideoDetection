import os, json
import numpy as np
import pandas as pd

base_dir = "/hpc2hdd/home/lsheng847/1122321/FakeVideoDetection"
data_path = "DFDC/white_audio_0_video_0_vsr"  # 按需修改
fp = os.path.join(base_dir, "infer", f"{data_path}.json")
check_metric = "wer"  # 按需修改
LOWER_IS_BETTER = {"wer", "cer", "loss", "distance"}

def load_infer(fp: str) -> pd.DataFrame:
    # 先尝试标准 json
    try:
        with open(fp, "r") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            df = pd.DataFrame(obj)
        elif isinstance(obj, dict):
            # 以文件路径为键的情况
            df = pd.DataFrame.from_dict(obj, orient="index").reset_index().rename(columns={"index": "file"})
        else:
            raise ValueError(f"Unexpected JSON type: {type(obj)}")
        return df
    except Exception:
        # 回退为 jsonlines
        df = pd.read_json(fp, lines=True)
        # 如果还是没有正常列，尝试再读一次并转置
        if df.shape[0] <= 2 and df.shape[1] > 2 and all(str(c).endswith(".mp4") for c in df.columns[:min(5, len(df.columns))]):
            df = pd.read_json(fp).T.reset_index().rename(columns={"index": "file"})
        return df

def ensure_label(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns.astype(str))
    # 直接存在
    for name in ["label", "y_true", "gt", "target", "is_fake", "fake"]:
        if name in cols:
            if name != "label":
                df = df.rename(columns={name: "label"})
            return df
    # 从类型字段推断：非 RR 记为假(1)，RR 为真(0)
    for name in ["pair_type", "type", "pair", "category_type"]:
        if name in cols:
            t = df[name].astype(str)
            df["label"] = (~t.str.contains("RealVideo-RealAudio|RR", case=False)).astype(int)
            return df
    # 若无法推断，打印可见列，抛错
    print("可用列:", df.columns.tolist()[:50])
    raise KeyError("缺少 label 列，且无法从 type/pair_type 推断，请检查数据格式。")

def prepare_score(df: pd.DataFrame, metric: str) -> np.ndarray:
    if metric not in df.columns:
        # 常见备选名
        aliases = {
            "cos": ["cos", "cosine", "cos_sim", "similarity"],
            "wer": ["wer", "word_error_rate"],
            "cer": ["cer", "char_error_rate"],
            "distance": ["dist", "distance", "l2", "euclidean"],
        }
        for k, alts in aliases.items():
            if metric == k:
                for a in alts:
                    if a in df.columns:
                        metric = a
                        break
        if metric not in df.columns:
            print("可用列:", df.columns.tolist()[:50])
            raise KeyError(f"未找到指标列: {metric}")
    s = df[metric].astype(float).values
    if metric in LOWER_IS_BETTER:
        s = -s  # 反向为“越大越假”
    return s

def print_stats(y, s, name="overall"):
    print(f"[{name}] n={len(y)} pos={int(y.sum())} neg={len(y)-int(y.sum())} "
          f"score_mean={s.mean():.4f} score_std={s.std():.4f}")

if __name__ == "__main__":
    df = load_infer(fp)
    if "label" not in df.columns and all(str(c).endswith(".mp4") for c in df.columns[:min(5, len(df.columns))]):
        df = pd.read_json(fp).T.reset_index().rename(columns={"index": "file"})
    df = ensure_label(df)

    y = df["label"].astype(int).values
    raw = df[check_metric].astype(float).values

    s_inv = -raw if check_metric in LOWER_IS_BETTER else raw
    from sklearn.metrics import roc_auc_score
    def safe_auc(y_true, y_score):
        if len(np.unique(y_true)) < 2 or np.isclose(np.std(y_score), 0): return float("nan")
        return roc_auc_score(y_true, y_score)

    print("[raw metric] mean_pos, mean_neg:", raw[y==1].mean(), raw[y==0].mean())
    print("[inverted]   mean_pos, mean_neg:", s_inv[y==1].mean(), s_inv[y==0].mean())
    print("AUC(raw)=", safe_auc(y, raw), " AUC(inverted)=", safe_auc(y, s_inv))

    # 选择更优方向继续统计
    use_inv = (safe_auc(y, s_inv) >= safe_auc(y, raw))
    s = s_inv if use_inv else raw
    print("Use inverted:", use_inv)
    print_stats(y, s, "overall")
    print("Overall AUC:", safe_auc(y, s))

    for cat_col, cats in [("pair_type", ["FF","FR","RF","RR"]), ("type", ["FakeVideo-FakeAudio","FakeVideo-RealAudio","RealVideo-FakeAudio","RealVideo-RealAudio"])]:
        if cat_col in df.columns:
            for c in cats:
                sub = df[df[cat_col] == c]
                if len(sub)==0: continue
                ys = sub["label"].astype(int).values
                rs = sub[check_metric].astype(float).values
                ss = (-rs if check_metric in LOWER_IS_BETTER else rs)
                if not use_inv: ss = rs
                print_stats(ys, ss, c)
                print(f"{c} AUC:", safe_auc(ys, ss))