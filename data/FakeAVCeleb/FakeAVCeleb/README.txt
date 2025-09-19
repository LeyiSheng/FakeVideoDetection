
We used 4 different deepfake generation and voice synthesis methods to generate our FakeAVCeleb. 
To generate deepfake videos, we used recent most popular methods: Faceswap, FSGAN, and Wav2Lip. 
For generating cloned audio, we used a tool called Real time voice cloning (SV2TTS). 
These methods are publicly available and can be downloaded from the urls mentioned below:
Faceswap: https://github.com/deepfakes/faceswap
FSGAN: https://github.com/YuvalNirkin/fsgan
Wav2Lip: https://github.com/Rudrabha/Wav2Lip
RTVC (SV2TTS): https://github.com/CorentinJ/Real-Time-Voice-Cloning


The directory structure is divided into four parent directories/categories:

A > RealVideo-RealAudio  (Real Voxceleb videos)
B > RealVideo-FakeAudio  (Deepfake)
C > FakeVideo-RealAudio  (Deepfake)
D > FakeVideo-FakeAudio  (Deepfake)

A metadata dile is also provided with the dataset which contains following labels:

Label:		Description:						Example:
source		Source video file id				id00166
target1		Target 1 video file id				id01637
target2		Target 2 video file id				-
method		Method used to generate deepfake	faceswap
category	Deepfake category					C
type		Description of category				FakeVideo-RealAudio
gender		Gender of the person in video		Black
race		Race of the person in video			men
filename	Name of the .mp4 file				00010_id01637_5VjcPZm8knM_faceswap.mp4
path		Path of the video file				FakeVideo-RealAudio/Black/men/id00166	

