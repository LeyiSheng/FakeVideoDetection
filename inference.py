import os
import cv2
import sys
import json
import torch
import librosa
import hashlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

import torch.nn as nn

# =============================================================
# 配置类
# =============================================================
class InferenceConfig:
    # 模型相关配置
    VISUAL_MODEL_NAME = "MahmoudWSegni/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12"
    AUDIO_EMOTION_MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    
    # 特征提取配置
    NUM_FRAMES = 7
    FRAME_SIZE = (224, 224)
    TARGET_SR = 16000
    
    # 路径配置
    SAVED_MODEL_PATH = "./feature_cache_v1/best_model.pt"  # 训练保存的模型路径
    FEATURE_CACHE_DIR = "./inference_feature_cache"  # 推理时的特征缓存目录
    OUTPUT_DIR = "./inference_results"  # 结果保存目录
    
    # 批处理配置
    BATCH_SIZE = 32
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self):
        print(f"Using device: {self.DEVICE}")
        Path(self.FEATURE_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================
# 工具函数（从原代码复制）
# =============================================================
def sha1_of_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def safe_makedirs(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# =============================================================
# 视频帧和音频处理（从原代码复制）
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
# MLP分类器（从原代码复制）
# =============================================================
class MLPDetector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5), 
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================
# 推理类
# =============================================================
class DeepfakeInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.models = self._load_pretrained_models()
        self.classifier = self._load_classifier()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def _load_pretrained_models(self):
        """加载预训练的视觉和音频模型"""
        print("Loading visual emotion model...")
        visual_processor = AutoImageProcessor.from_pretrained(self.config.VISUAL_MODEL_NAME)
        visual_model = AutoModelForImageClassification.from_pretrained(
            self.config.VISUAL_MODEL_NAME
        ).to(self.config.DEVICE)
        visual_model.eval()
        
        print("Loading audio emotion model...")
        audio_feat = Wav2Vec2FeatureExtractor.from_pretrained(self.config.AUDIO_EMOTION_MODEL_NAME)
        audio_model = AutoModelForAudioClassification.from_pretrained(
            self.config.AUDIO_EMOTION_MODEL_NAME
        ).to(self.config.DEVICE)
        audio_model.eval()
        
        return {
            "visual_processor": visual_processor, 
            "visual_model": visual_model,
            "audio_feature": audio_feat, 
            "audio_model": audio_model
        }
    
    def _load_classifier(self):
        """加载训练好的MLP分类器"""
        # 计算特征维度
        V = int(getattr(self.models["visual_model"].config, "num_labels", 8))
        A = int(getattr(self.models["audio_model"].config, "num_labels", 3))
        feature_dim = V + 3 + 13 + A  # visual_logits + visual_metrics + mfcc + audio_logits
        
        print(f"Feature dimension: {feature_dim}")
        
        # 创建模型并加载权重
        model = MLPDetector(input_dim=feature_dim).to(self.config.DEVICE)
        
        if not os.path.exists(self.config.SAVED_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {self.config.SAVED_MODEL_PATH}")
        
        print(f"Loading classifier from {self.config.SAVED_MODEL_PATH}")
        model.load_state_dict(
            torch.load(self.config.SAVED_MODEL_PATH, map_location=self.config.DEVICE, weights_only=True)
        )
        model.eval()
        
        return model
    
    def extract_visual_features(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """提取视频的视觉特征"""
        frames = extract_frames_uniform(video_path, self.config.NUM_FRAMES)
        if len(frames) == 0:
            raise RuntimeError(f"No frames extracted: {video_path}")
        
        proc = self.models["visual_processor"]
        vmodel = self.models["visual_model"]
        
        logits_list, metrics_list = [], []
        for fr in frames:
            face = detect_and_crop_face(fr, self.face_cascade, target_size=self.config.FRAME_SIZE)
            metrics_list.append(calc_visual_metrics(face))
            inputs = proc(images=face, return_tensors="pt")
            inputs = {k: v.to(self.config.DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                out = vmodel(**inputs)
            logits_list.append(out.logits.squeeze().cpu().numpy())
        
        v_logits_mean = np.mean(np.stack(logits_list, axis=0), axis=0)
        v_metrics_mean = np.mean(np.stack(metrics_list, axis=0), axis=0)
        return v_logits_mean, v_metrics_mean
    
    def extract_audio_features(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """提取视频的音频特征"""
        tmp_dir = os.path.join(self.config.FEATURE_CACHE_DIR, "_tmp_wav")
        wav_path = ensure_audio_wav(video_path, tmp_dir, sr=self.config.TARGET_SR)
        audio, sr = librosa.load(wav_path, sr=self.config.TARGET_SR)
        
        # MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # 音频情感特征
        afe, amodel = self.models["audio_feature"], self.models["audio_model"]
        inputs = afe(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.config.DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = amodel(**inputs)
        a_logits = out.logits.squeeze().cpu().numpy()
        
        return mfcc_mean, a_logits
    
    def extract_features(self, video_path: str) -> np.ndarray:
        """提取完整的特征向量"""
        # 检查缓存
        cache_path = os.path.join(
            self.config.FEATURE_CACHE_DIR, 
            f"{sha1_of_text(video_path)}.npz"
        )
        
        if os.path.exists(cache_path):
            print(f"Loading cached features for {os.path.basename(video_path)}")
            data = np.load(cache_path)
            feature_vector = np.concatenate([
                data["visual_logits"],
                data["visual_metrics"],
                data["mfcc"],
                data["audio_logits"]
            ], axis=0).astype(np.float32)
        else:
            print(f"Extracting features for {os.path.basename(video_path)}")
            v_logits, v_metrics = self.extract_visual_features(video_path)
            mfcc, a_logits = self.extract_audio_features(video_path)
            
            # 保存缓存
            np.savez_compressed(
                cache_path,
                visual_logits=v_logits.astype(np.float32),
                visual_metrics=v_metrics.astype(np.float32),
                mfcc=mfcc.astype(np.float32),
                audio_logits=a_logits.astype(np.float32),
                video_path=video_path
            )
            
            feature_vector = np.concatenate([v_logits, v_metrics, mfcc, a_logits], axis=0).astype(np.float32)
        
        return feature_vector
    
    def predict_single(self, video_path: str) -> Dict:
        """预测单个视频"""
        try:
            # 提取特征
            features = self.extract_features(video_path)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.config.DEVICE)
            
            # 预测
            with torch.no_grad():
                logits = self.classifier(features_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(logits, dim=1).item()
                confidence = probs[0, pred_class].item()
            
            result = {
                "video_path": video_path,
                "prediction": "real" if pred_class == 1 else "fake",
                "prediction_class": pred_class,
                "confidence": confidence,
                "fake_probability": probs[0, 0].item(),
                "real_probability": probs[0, 1].item(),
                "status": "success"
            }
        except Exception as e:
            result = {
                "video_path": video_path,
                "prediction": None,
                "prediction_class": None,
                "confidence": None,
                "fake_probability": None,
                "real_probability": None,
                "status": "failed",
                "error": str(e)
            }
        
        return result
    
    def predict_batch(self, video_paths: List[str]) -> List[Dict]:
        """批量预测多个视频"""
        results = []
        for video_path in tqdm(video_paths, desc="Processing videos"):
            if not os.path.exists(video_path):
                print(f"[WARNING] Video not found: {video_path}")
                results.append({
                    "video_path": video_path,
                    "status": "failed",
                    "error": "File not found"
                })
                continue
            
            result = self.predict_single(video_path)
            results.append(result)
            
            # 打印结果
            if result["status"] == "success":
                print(f"  → {result['prediction'].upper()} (confidence: {result['confidence']:.2%})")
            else:
                print(f"  → Failed: {result['error']}")
        
        return results
    
    def predict_from_directory(self, directory: str, extensions: List[str] = None) -> List[Dict]:
        """预测目录中的所有视频"""
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        video_paths = []
        for ext in extensions:
            video_paths.extend(Path(directory).rglob(f"*{ext}"))
        
        video_paths = [str(p) for p in video_paths]
        print(f"Found {len(video_paths)} videos in {directory}")
        
        return self.predict_batch(video_paths)
    
    def predict_from_json(self, json_path: str, video_base_dir: str = None) -> List[Dict]:
        """从JSON文件加载视频路径并预测"""
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        video_paths = []
        for video_name in metadata.keys():
            if video_base_dir:
                video_path = os.path.join(video_base_dir, video_name)
            else:
                video_path = video_name
            video_paths.append(video_path)
        
        print(f"Loaded {len(video_paths)} videos from {json_path}")
        return self.predict_batch(video_paths)
    
    def save_results(self, results: List[Dict], output_file: str = None):
        """保存预测结果"""
        if output_file is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.config.OUTPUT_DIR, f"predictions_{timestamp}.csv")
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存CSV
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # 打印统计信息
        if len(results) > 0:
            success_results = [r for r in results if r["status"] == "success"]
            if success_results:
                fake_count = sum(1 for r in success_results if r["prediction"] == "fake")
                real_count = sum(1 for r in success_results if r["prediction"] == "real")
                print(f"\n=== Summary ===")
                print(f"Total videos: {len(results)}")
                print(f"Successfully processed: {len(success_results)}")
                print(f"Failed: {len(results) - len(success_results)}")
                print(f"Predicted as FAKE: {fake_count} ({fake_count/len(success_results):.1%})")
                print(f"Predicted as REAL: {real_count} ({real_count/len(success_results):.1%})")


# =============================================================
# 主函数示例
# =============================================================
def main():
    """主函数示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("--mode", type=str, required=True, 
                      choices=["single", "batch", "directory", "json"],
                      help="Inference mode")
    parser.add_argument("--input", type=str, required=True,
                      help="Input video path, directory, or JSON file")
    parser.add_argument("--video-base-dir", type=str, default=None,
                      help="Base directory for videos (for JSON mode)")
    parser.add_argument("--model-path", type=str, default=None,
                      help="Path to saved model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                      help="Output CSV file path")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size for processing")
    
    args = parser.parse_args()
    
    # 初始化配置
    config = InferenceConfig()
    if args.model_path:
        config.SAVED_MODEL_PATH = args.model_path
    config.BATCH_SIZE = args.batch_size
    
    # 初始化推理器
    print("Initializing deepfake detector...")
    detector = DeepfakeInference(config)
    
    # 执行推理
    if args.mode == "single":
        # 单个视频推理
        result = detector.predict_single(args.input)
        print("\n=== Result ===")
        if result["status"] == "success":
            print(f"Video: {result['video_path']}")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Fake probability: {result['fake_probability']:.2%}")
            print(f"Real probability: {result['real_probability']:.2%}")
        else:
            print(f"Failed to process: {result['error']}")
        
        # 保存结果
        if args.output:
            detector.save_results([result], args.output)
    
    elif args.mode == "batch":
        # 批量视频推理（从文本文件读取路径列表）
        with open(args.input, 'r') as f:
            video_paths = [line.strip() for line in f if line.strip()]
        results = detector.predict_batch(video_paths)
        detector.save_results(results, args.output)
    
    elif args.mode == "directory":
        # 目录推理
        results = detector.predict_from_directory(args.input)
        detector.save_results(results, args.output)
    
    elif args.mode == "json":
        # JSON文件推理
        results = detector.predict_from_json(args.input, args.video_base_dir)
        detector.save_results(results, args.output)


def example_usage():
    """使用示例（不需要命令行参数）"""
    # 初始化配置
    config = InferenceConfig()
    
    # 如果模型保存在其他位置，可以修改路径
    # config.SAVED_MODEL_PATH = "/path/to/your/best_model.pt"
    
    # 初始化推理器
    detector = DeepfakeInference(config)
    
    # 示例1：预测单个视频
    print("\n=== Example 1: Single Video ===")
    video_path = "/path/to/your/test_video.mp4"
    if os.path.exists(video_path):
        result = detector.predict_single(video_path)
        print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2%}")
    
    # 示例2：批量预测
    print("\n=== Example 2: Batch Prediction ===")
    video_list = [
        "/path/to/video1.mp4",
        "/path/to/video2.mp4",
        "/path/to/video3.mp4"
    ]
    # 过滤存在的视频
    video_list = [v for v in video_list if os.path.exists(v)]
    if video_list:
        results = detector.predict_batch(video_list)
        detector.save_results(results)
    
    # 示例3：预测整个目录
    print("\n=== Example 3: Directory Prediction ===")
    test_dir = "/path/to/test/directory"
    if os.path.exists(test_dir):
        results = detector.predict_from_directory(test_dir)
        detector.save_results(results)
    
    # 示例4：从JSON文件预测
    print("\n=== Example 4: JSON Prediction ===")
    json_path = "/path/to/metadata.json"
    video_base = "/path/to/videos"
    if os.path.exists(json_path):
        results = detector.predict_from_json(json_path, video_base)
        detector.save_results(results)


if __name__ == "__main__":
    # 检查环境
    print("CUDA available:", torch.cuda.is_available())
    
    # 如果有命令行参数则使用main()，否则运行示例
    if len(sys.argv) > 1:
        main()
    else:
        print("\nNo command line arguments provided. Running example usage...")
        print("For command line usage, run with --help flag")
        example_usage()