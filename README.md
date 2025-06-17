# Zero-Shot Fake Video Detection by Audio-Visual Consistency

## Introduction

This repository provides the implementation for [Zero-Shot Fake Video Detection by Audio-Visual Consistency](https://arxiv.org/abs/2406.07854), a content-consistency based method for detecting fake videos. Our approach leverages the **FakeAVCeleb** and **DFDC** datasets and builds upon the [auto-avsr](https://github.com/mpc001/auto_avsr/tree/main) framework.

---

## Data Preparation

To get started, you'll need to prepare your datasets:

1.  **Download Datasets:**
    * [FakeAVCeleb](https://sites.google.com/view/fakeavcelebdash-lab/)
    * [DFDC (DeepFake Detection Challenge)](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)

2.  **Pre-processing:**
    Our pre-processing pipeline, adapted from `auto-avsr`, ensures consistent data formatting:
    * Videos are converted to **25 FPS**.
    * Audio is converted to **16 kHz mono**.
    * The speaker's lip region is detected and cropped from each video frame.
    * Cropped frames are resized to a uniform **96x96 pixels**.
    * For a detailed look at the complete pre-processing steps, refer to the [auto-avsr preparation guide](https://github.com/mpc001/auto_avsr/tree/main/preparation).

3.  **Create CSV File List:**
    After pre-processing, create a CSV file (e.g., `data/your_dataset.csv`) with the following format:
    `absolute_video_file_path, video_frames, segment_label, audio_label, video_label`

    **Example:**
    `/your_path/FakeAVCeleb/video/FakeVideo-FakeAudio/African/men/id00366/00118_id00076_Isiq7cA-DNE_faceswap_id01170_wavtolip.mp4, 148, 0, 0, 0`

    For convenience, we've already prepared file lists for `FakeAVCeleb` and `DFDC` in the `data` folder.

---

## Setup

Follow these steps to set up your environment and download necessary models:

1.  **Create Environment:**

    ```bash
    conda create -y -n fakevideodetection python=3.10
    conda activate fakevideodetection
    pip install -r requirements.txt
    ```

2.  **Download Pre-trained Models:**
    Download the following pre-trained models from [VSR for ML](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) and place them in the `pretrained_model` folder:

    | Component       | WER  | URL                                                                   | Size (MB) |
    | :-------------- | :--- | :-------------------------------------------------------------------- | :-------- |
    | **Visual-only** | 19.1 | [GoogleDrive](http://bit.ly/40EAtyX) or [BaiduDrive](https://bit.ly/3ZjbrV5) (key: dqsy) | 891       |
    | **Audio-only** | 1.0  | [GoogleDrive](http://bit.ly/3ZSdh0l) or [BaiduDrive](http://bit.ly/3Z1TlGU) (key: dvf2) | 860       |

---

## Inference

Once everything is set up, you can run the inference:

Configure your settings in `run.sh` and then execute:

```bash
bash run.sh