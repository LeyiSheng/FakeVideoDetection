import os
import cv2
import sys
import math
import json
import time
import torch
import librosa
import random
import hashlib
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix

from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================
# 0) 实用工具
# =============================================================

def set_seed(seed: int = 101):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_makedirs(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def sha1_of_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# =============================================================
# 1) 配置
# =============================================================
class Config:
    DATASET_PATH = "/hpc2hdd/home/lsheng847/1122321/FakeVideoDetection/data/FakeAVCeleb"
    LABEL_PATH = "/hpc2hdd/home/lsheng847/1122321/FakeVideoDetection/data/FakeAVCeleb/label.csv"
    FEATURE_DIR = "./feature_cache_v1"

    VISUAL_MODEL_NAME = "MahmoudWSegni/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12"
    AUDIO_EMOTION_MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    NUM_FRAMES = 7
    FRAME_SIZE = (224, 224)

    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    SEED = 101
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_SR = 16000
    DISABLE_AUDIO = True  # audio features disabled (set to False to re-enable)

    def __init__(self):
        set_seed(self.SEED)
        print(f"Using device: {self.DEVICE}")


config = Config()


# =============================================================
# 2) 数据加载
# =============================================================
import csv

def filelist(listcsv):
    fns = []
    with open(listcsv) as fp:
        lines = fp.readlines()
    for line in lines:
        fn, length, label, audio_label, video_label = line.split(',')
        if os.path.exists(fn):
            fns.append((fn.strip(), length.strip(), label.strip(), audio_label.strip(), video_label.strip()))
    return fns

def filelist_from_json(json_path, video_base_dir):
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    fns = []
    for video_name, info in metadata.items():
        # 拼接完整路径
        video_path = os.path.join(video_base_dir, video_name)
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"[WARN] Video file not found: {video_path}")
            continue  # 跳过不存在的文件
        
        label_str = info.get("label", "fake").lower()
        if label_str == "real":
            overall_label = 1
        else:
            overall_label = 0
        
        fns.append((video_path, str(overall_label)))
    return fns

# =============================================================
# 3) 视频帧与音频
# =============================================================

def extract_frames_uniform(video_path: str, num_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total - 1), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def detect_and_crop_face(frame: np.ndarray, face_cascade=None, target_size=(224, 224)) -> np.ndarray:
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda it: it[2] * it[3])
        pad = int(0.2 * h)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)
        face = frame[y:y + h, x:x + w]
    else:
        face = frame
    return cv2.resize(face, target_size, interpolation=cv2.INTER_LINEAR)


def calc_visual_metrics(frame_rgb: np.ndarray) -> List[float]:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0)) / (edges.shape[0] * edges.shape[1])
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = float(np.mean(np.abs(gray.astype(np.float32) - blurred.astype(np.float32))))
    return [sharpness, edge_density, noise]


def extract_audio_ffmpeg(video_path: str, wav_path: str, sr: int = 16000) -> bool:
    try:
        import subprocess
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), wav_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0
    except Exception:
        return False


def ensure_audio_wav(video_path: str, tmp_dir: str, sr: int = 16000) -> str:
    safe_makedirs(tmp_dir)
    stem = sha1_of_text(video_path)
    wav_path = os.path.join(tmp_dir, f"{stem}.wav")
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return wav_path
    ok = extract_audio_ffmpeg(video_path, wav_path, sr)
    if not ok:
        raise RuntimeError(f"Audio extraction failed for {video_path}. Please install ffmpeg.")
    return wav_path


# =============================================================
# 4) 加载模型
# =============================================================

def load_models():
    print("Loading visual emotion model …")
    visual_processor = AutoImageProcessor.from_pretrained(config.VISUAL_MODEL_NAME)
    visual_model = AutoModelForImageClassification.from_pretrained(config.VISUAL_MODEL_NAME).to(config.DEVICE)
    visual_model.eval()

    if config.DISABLE_AUDIO:
        audio_feat = None
        audio_model = None
        print("Audio features disabled; skipping audio model load.")
    else:
        print("Loading audio emotion model …")
        audio_feat = Wav2Vec2FeatureExtractor.from_pretrained(config.AUDIO_EMOTION_MODEL_NAME)
        audio_model = AutoModelForAudioClassification.from_pretrained(config.AUDIO_EMOTION_MODEL_NAME).to(config.DEVICE)
        audio_model.eval()

    return {"visual_processor": visual_processor, "visual_model": visual_model,
            "audio_feature": audio_feat, "audio_model": audio_model}


# =============================================================
# 5) 特征抽取
# =============================================================

def extract_visual_feature_for_video(video_path: str, models) -> Tuple[np.ndarray, np.ndarray]:
    frames = extract_frames_uniform(video_path, config.NUM_FRAMES)
    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted: {video_path}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    proc = models["visual_processor"]
    vmodel = models["visual_model"]

    logits_list, metrics_list = [], []
    for fr in frames:
        face = detect_and_crop_face(fr, face_cascade, target_size=config.FRAME_SIZE)
        metrics_list.append(calc_visual_metrics(face))
        inputs = proc(images=face, return_tensors="pt")
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = vmodel(**inputs)
        logits_list.append(out.logits.squeeze().cpu().numpy())

    v_logits_mean = np.mean(np.stack(logits_list, axis=0), axis=0)
    v_metrics_mean = np.mean(np.stack(metrics_list, axis=0), axis=0)
    return v_logits_mean, v_metrics_mean


def extract_audio_feature_for_video(video_path: str, models) -> Tuple[np.ndarray, np.ndarray]:
    if config.DISABLE_AUDIO:
        audio_model = models.get("audio_model") if isinstance(models, dict) else None
        num_labels = int(getattr(getattr(audio_model, "config", None), "num_labels", 3))
        return np.zeros(13, dtype=np.float32), np.zeros(num_labels, dtype=np.float32)

    tmp_dir = os.path.join(config.FEATURE_DIR, "_tmp_wav")
    wav_path = ensure_audio_wav(video_path, tmp_dir, sr=config.TARGET_SR)
    audio, sr = librosa.load(wav_path, sr=config.TARGET_SR)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    afe, amodel = models["audio_feature"], models["audio_model"]
    inputs = afe(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = amodel(**inputs)
    a_logits = out.logits.squeeze().cpu().numpy()
    return mfcc_mean, a_logits


def build_feature_once(video_path: str, models) -> Dict[str, np.ndarray]:
    v_logits, v_metrics = extract_visual_feature_for_video(video_path, models)
    mfcc, a_logits = extract_audio_feature_for_video(video_path, models)
    return {"visual_logits": v_logits.astype(np.float32),
            "visual_metrics": v_metrics.astype(np.float32),
            "mfcc": mfcc.astype(np.float32),
            "audio_logits": a_logits.astype(np.float32)}


def feature_vector_from_dict(d: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([d["visual_logits"], d["visual_metrics"], d["mfcc"], d["audio_logits"]], axis=0).astype(np.float32)


def feature_dim_from_models(models) -> int:
    V = int(getattr(models["visual_model"].config, "num_labels", 8))
    audio_model = models.get("audio_model") if isinstance(models, dict) else None
    A = int(getattr(getattr(audio_model, "config", None), "num_labels", 3))
    return V + 3 + 13 + A


def cache_path_for_video(video_path: str) -> str:
    return os.path.join(config.FEATURE_DIR, f"{sha1_of_text(video_path)}.npz")


def build_feature_cache(video_paths: List[str], models) -> None:
    safe_makedirs(config.FEATURE_DIR)
    for vp in tqdm(video_paths, desc="Building feature cache"):
        outp = cache_path_for_video(vp)
        if os.path.exists(outp) and os.path.getsize(outp) > 0:
            continue
        try:
            feat = build_feature_once(vp, models)
            np.savez_compressed(outp, **feat, video_path=vp)
        except Exception as e:
            print(f"[WARN] feature extraction failed for {vp}: {e}")


# =============================================================
# 6) 数据集
# =============================================================
class FeatureDataset(Dataset):
    def __init__(self, video_paths: List[str], overall_labels: List[int]):
        self.video_paths = video_paths
        self.labels = overall_labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        vp = self.video_paths[idx]
        cache_fp = cache_path_for_video(vp)
        if not os.path.exists(cache_fp):
            raise RuntimeError(f"Missing cached feature for {vp}")
        data = np.load(cache_fp)
        feat = feature_vector_from_dict({
            "visual_logits": data["visual_logits"],
            "visual_metrics": data["visual_metrics"],
            "mfcc": data["mfcc"],
            "audio_logits": data["audio_logits"],
        })
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def create_loader(video_paths, overall_labels, batch_size, shuffle):
    num_workers = min(4, os.cpu_count() or 2)
    pin = config.DEVICE.type == "cuda"
    ds = FeatureDataset(video_paths, overall_labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)


# =============================================================
# 7) 分类器
# =============================================================
class MLPDetector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================
# 8) 训练与评估
# =============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for feats, labels in loader:
        feats, labels = feats.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


def evaluate(model, loader, detailed: bool = False):
    model.eval()
    all_preds, all_labels, all_probs, total = [], [], [], 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(config.DEVICE), labels.to(config.DEVICE)
            logits = model(feats)
            loss = criterion(logits, labels)
            total += float(loss.item())
            
            # 获取预测类别
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            # 获取预测概率（用于AUC计算）
            probs = torch.softmax(logits, dim=1)[:, 1]  # 取正类(1)的概率
            all_probs.append(probs.cpu())
            
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # 计算各种指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")
    
    # 计算AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError as e:
        # 如果只有一个类别，AUC无法计算
        print(f"[WARN] AUC calculation failed: {e}")
        auc = 0.0
    
    if detailed:
        print("Classification report:\n", classification_report(all_labels, all_preds, digits=4))
        print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds))
        print(f"AUC Score: {auc:.4f}")
    
    return total / max(1, len(loader)), acc, f1, rec, auc


# =============================================================
# 9) 主流程
# =============================================================
# =============================================================
def class_dist(name, labels):
    import numpy as np
    arr = np.array(labels, dtype=int)
    total = arr.size
    pos = int((arr == 1).sum())
    neg = total - pos
    print(f"[{name}] total={total} neg(0)={neg} pos(1)={pos}")

def main():
    import os, shutil, sys
    print("python:", sys.executable)
    print("PATH :", os.environ.get("PATH"))
    print("which ffmpeg (shutil):", shutil.which("ffmpeg"))
    # 使用filelist函数加载数据
    fns = filelist(config.LABEL_PATH)
    
    if len(fns) == 0:
        print("[ERROR] No samples loaded.")
        return
    
    # 解析fns中的数据
    video_paths = [fn[0] for fn in fns]
    lengths = [fn[1] for fn in fns]
    overall_labels = [int(fn[2]) for fn in fns]
    audio_labels   = [int(fn[3]) for fn in fns]
    video_labels   = [int(fn[4]) for fn in fns]
    print(f"Loaded {len(video_paths)} samples.")
    class_dist("ALL-overall", overall_labels)
    class_dist("ALL-audio",   audio_labels)
    class_dist("ALL-video",   video_labels)
    
    # 划分训练集、验证集和测试集
    train_val_paths, test_paths, train_val_lengths, test_lengths, \
    train_val_audio, test_audio, train_val_video, test_video, \
    train_val_overall, test_overall = train_test_split(
        video_paths, lengths, audio_labels, video_labels, overall_labels, 
        test_size=config.TEST_SIZE, random_state=config.SEED, stratify=overall_labels
    )
    
    train_paths, val_paths, train_lengths, val_lengths, \
    train_audio, val_audio, train_video, val_video, \
    train_overall, val_overall = train_test_split(
        train_val_paths, train_val_lengths, train_val_audio, train_val_video, train_val_overall,
        test_size=config.VAL_SIZE / (1 - config.TEST_SIZE), 
        random_state=config.SEED, stratify=train_val_overall
    )
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    class_dist("Train", train_overall)
    class_dist("Val",   val_overall)
    class_dist("Test",  test_overall)

    # 加载模型并构建特征缓存
    models = load_models()
    build_feature_cache(train_paths + val_paths + test_paths, models)

    # 创建数据加载器x
    feat_dim = feature_dim_from_models(models)
    train_loader = create_loader(train_paths, train_overall, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = create_loader(val_paths, val_overall, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = create_loader(test_paths, test_overall, batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化分类器
    print(f"Feature dimension: {feat_dim}")
    model = MLPDetector(input_dim=feat_dim).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, val_rec, val_auc = evaluate(model, val_loader)  # 添加val_auc

        print(f"Epoch {epoch+1:2d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Rec: {val_rec:.4f}, AUC: {val_auc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.FEATURE_DIR, "best_model.pt"))
            print(f"  [+] Best model saved with val acc: {val_acc:.4f}")

    # 加载最佳模型并进行测试
    print("\nLoading best model for final test...")
    model.load_state_dict(torch.load(os.path.join(config.FEATURE_DIR, "best_model.pt"), map_location=config.DEVICE))
    test_loss, test_acc, test_f1, test_rec, test_auc = evaluate(model, test_loader, detailed=True)  # 添加test_auc

    print(f"\nFinal Test Results | Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Rec: {test_rec:.4f}, AUC: {test_auc:.4f}")

def main_json():
    """
    使用 JSON 格式数据集的主函数
    """
    import os, shutil, sys
    print("python:", sys.executable)
    print("PATH :", os.environ.get("PATH"))
    print("which ffmpeg (shutil):", shutil.which("ffmpeg"))
    
    # JSON数据集配置
    json_path = "/hpc2hdd/home/lsheng847/1122321/FakeVideoDetection/data/FMFCC-V-Competition/metadata.json"
    video_base_dir = "/hpc2hdd/home/lsheng847/1122321/FakeVideoDetection/data/FMFCC-V-Competition"
    
    # 使用新的JSON加载函数
    fns = filelist_from_json(json_path, video_base_dir)
    
    if len(fns) == 0:
        print("[ERROR] No samples loaded from JSON.")
        return
    
    # 解析fns中的数据（与原来的main函数相同）
    video_paths = [fn[0] for fn in fns]
    overall_labels = [int(fn[1]) for fn in fns]
    print(f"Loaded {len(video_paths)} samples from JSON.")
    class_dist("ALL-overall", overall_labels)
    
    # 划分训练集、验证集和测试集
    train_val_paths, test_paths, train_val_overall, test_overall = train_test_split(
        video_paths, overall_labels, 
        test_size=config.TEST_SIZE, random_state=config.SEED, stratify=overall_labels
    )
    
    train_paths, val_paths, train_overall, val_overall = train_test_split(
        train_val_paths, train_val_overall,
        test_size=config.VAL_SIZE / (1 - config.TEST_SIZE), 
        random_state=config.SEED, stratify=train_val_overall
    )
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    class_dist("Train", train_overall)
    class_dist("Val",   val_overall)
    class_dist("Test",  test_overall)

    # 加载模型并构建特征缓存
    models = load_models()
    build_feature_cache(train_paths + val_paths + test_paths, models)

    # 创建数据加载器
    feat_dim = feature_dim_from_models(models)
    train_loader = create_loader(train_paths, train_overall, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = create_loader(val_paths, val_overall, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = create_loader(test_paths, test_overall, batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化分类器
    print(f"Feature dimension: {feat_dim}")
    model = MLPDetector(input_dim=feat_dim).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, val_rec, val_auc = evaluate(model, val_loader)

        import pynvml  # pip install pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"GPU utilization: {info.gpu}%")
        print(f"Epoch {epoch+1:2d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Rec: {val_rec:.4f}, AUC: {val_auc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.FEATURE_DIR, "best_model.pt"))
            print(f"  [+] Best model saved with val acc: {val_acc:.4f}")

    # 加载最佳模型并进行测试
    print("\nLoading best model for final test...")
    model.load_state_dict(torch.load(os.path.join(config.FEATURE_DIR, "best_model.pt"), map_location=config.DEVICE, weights_only=True))
    test_loss, test_acc, test_f1, test_rec, test_auc = evaluate(model, test_loader, detailed=True)

    print(f"\nFinal Test Results | Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Rec: {test_rec:.4f}, AUC: {test_auc:.4f}")

# 修改 __main__ 部分
if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", config.DEVICE)
    # 使用 CSV 数据集
    # main()
    
    # 使用 JSON 数据集
    main()

import pynvml  # pip install pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU utilization: {info.gpu}%")
