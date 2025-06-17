import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


@dataclass
class Config:
    save_path: str = 'LAV-DF/ctc_cos'
    data_path: str = 'Noisy/fakeavceleb_noisy_audio_0.5_video_0.5_vsr'
    check_metric: str = 'cos'
    base_dir: str = '/work/lixiaolou/program/FakeVideoDetection'
    
    @property
    def data_file(self) -> str:
        return f'{self.base_dir}/infer/{self.data_path}.json'
    
    @property
    def output_dir(self) -> str:
        return f'{self.base_dir}/pic/{self.save_path}'


@dataclass
class SubsetFilter:
    name: str
    video_label: Optional[str] = None
    audio_label: Optional[str] = None
    
    def matches(self, entry: Dict) -> bool:
        """Check if the data entry matches the filter conditions"""
        if entry['label'] == '1':  # always include positive examples
            return True
        
        video_match = self.video_label is None or entry.get('video label') == self.video_label
        audio_match = self.audio_label is None or entry.get('audio label') == self.audio_label
        
        return video_match and audio_match


class ROCAnalyzer:
    """ROC curve analyzer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data = self._load_data()
        
        # define subset filters
        self.subsets = {
            'FF': SubsetFilter('FF', video_label='0', audio_label='0'),
            'FR': SubsetFilter('FR', video_label='0', audio_label='1'),
            'RF': SubsetFilter('RF', video_label='1', audio_label='0'),
        }
    
    def _load_data(self) -> Dict:
        """Load JSON data"""
        with open(self.config.data_file, 'r') as file:
            return json.load(file)
    
    def _ensure_output_dir(self):
        """Ensure output directory exists"""
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _extract_labels_and_scores(self, data_filter=None) -> Tuple[List[int], List[float]]:
        """Extract labels and scores"""
        labels = []
        scores = []
        
        for entry in self.data.values():
            if data_filter is None or data_filter.matches(entry):
                labels.append(int(entry['label']))
                # reverse WER to score, and limit in 0-1 range
                score = 1 - min(entry[self.config.check_metric], 1)
                scores.append(score)
        
        return labels, scores
    
    def _extract_wer_by_label(self, data_filter=None) -> Tuple[List[float], List[float]]:
        """Extract WER values by label"""
        positive_wer = []
        negative_wer = []
        
        for entry in self.data.values():
            if data_filter is None or data_filter.matches(entry):
                wer_value = entry[self.config.check_metric]
                if int(entry['label']) == 1:
                    positive_wer.append(wer_value)
                else:
                    negative_wer.append(wer_value)
        
        return positive_wer, negative_wer
    
    def calculate_roc_metrics(self, data_filter=None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculate ROC metrics"""
        labels, scores = self._extract_labels_and_scores(data_filter)
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    
    def plot_single_roc(self, fpr: np.ndarray, tpr: np.ndarray, 
                       roc_auc: float, title: str = "ROC Curve") -> str:
        """Plot single ROC curve"""
        self._ensure_output_dir()
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        output_path = f'{self.config.output_dir}/roc_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_multiple_roc(self, roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]]) -> str:
        """Plot multiple ROC curves"""
        self._ensure_output_dir()
        
        plt.figure(figsize=(10, 8))
        colors = ['darkorange', 'red', 'green', 'blue', 'purple']
        
        for i, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{name} ROC curve (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        output_path = f'{self.config.output_dir}/roc_curves_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_wer_distribution(self, positive_wer: List[float], negative_wer: List[float], 
                             name: str = "WER") -> str:
        """Plot WER distribution"""
        self._ensure_output_dir()
        
        plt.figure(figsize=(10, 6))
        plt.hist(positive_wer, alpha=0.6, label='Positive (label=1)', 
                bins=20, density=True, color='skyblue', edgecolor='black')
        plt.hist(negative_wer, alpha=0.6, label='Negative (label=0)', 
                bins=20, density=True, color='lightcoral', edgecolor='black')
        plt.xlabel(f'{self.config.check_metric.upper()}')
        plt.ylabel('Density')
        plt.title(f'{name} Distribution for Positive and Negative Examples')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        output_path = f'{self.config.output_dir}/{name}_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def run_full_analysis(self):
        """Run full ROC analysis"""
        print("Starting ROC analysis...")
        
        # calculate overall ROC metrics
        overall_fpr, overall_tpr, overall_auc = self.calculate_roc_metrics()
        print(f'总体 AUC: {overall_auc:.4f}')
        
        # calculate ROC metrics for each subset
        subset_roc_data = {'Overall': (overall_fpr, overall_tpr, overall_auc)}
        
        for subset_name, subset_filter in self.subsets.items():
            fpr, tpr, roc_auc = self.calculate_roc_metrics(subset_filter)
            subset_roc_data[subset_name] = (fpr, tpr, roc_auc)
            print(f'{subset_name} AUC: {roc_auc:.4f}')
        
        # plot ROC curves
        print("\nGenerating ROC curve...")
        single_roc_path = self.plot_single_roc(overall_fpr, overall_tpr, overall_auc)
        print(f"Single ROC curve saved to: {single_roc_path}")
        
        multiple_roc_path = self.plot_multiple_roc(subset_roc_data)
        print(f"Multiple ROC curves saved to: {multiple_roc_path}")
        
        # plot WER distribution
        print("\nGenerating WER distribution...")
        
        # overall WER distribution
        overall_pos_wer, overall_neg_wer = self._extract_wer_by_label()
        overall_dist_path = self.plot_wer_distribution(overall_pos_wer, overall_neg_wer, "Overall_WER")
        print(f"Overall WER distribution saved to: {overall_dist_path}")
        
        # WER distribution for each subset
        for subset_name, subset_filter in self.subsets.items():
            pos_wer, neg_wer = self._extract_wer_by_label(subset_filter)
            if pos_wer and neg_wer:  # ensure data exists
                dist_path = self.plot_wer_distribution(pos_wer, neg_wer, f"{subset_name}_WER")
                print(f"{subset_name} WER distribution saved to: {dist_path}")
        
        print(f"\nAll analysis results saved to: {self.config.output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ROC analyzer')
    parser.add_argument('--save_path', type=str, default='FakeAVCeleb/', 
                       help='save result path (default: FakeAVCeleb/)')
    parser.add_argument('--data_path', type=str, default='FakeAVCeleb/white_audio_999999_video_0_vsr',
                       help='data file path (default: FakeAVCeleb/white_audio_999999_video_0_vsr)')
    parser.add_argument('--check_metric', type=str, default='wer', choices=['wer', 'cos', 'ctc'],
                       help='check metric type (default: wer, optional: wer, cos, ctc)')
    parser.add_argument('--base_dir', type=str, default='./FakeVideoDetection',
                       help='base directory path (default: ./FakeVideoDetection)')
    
    args = parser.parse_args()
    
    config = Config(
        save_path=args.save_path,
        data_path=args.data_path,
        check_metric=args.check_metric,
        base_dir=args.base_dir
    )
    
    print("  config parameters:")
    print(f"  save path: {config.save_path}")
    print(f"  data path: {config.data_path}")
    print(f"  check metric: {config.check_metric}")
    print(f"  base directory: {config.base_dir}")
    print(f"  data file: {config.data_file}")
    print(f"  output directory: {config.output_dir}")
    print("-" * 50)
    
    analyzer = ROCAnalyzer(config)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main() 