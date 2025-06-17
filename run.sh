audio_noisy=(999999)
video_noisy=(0)
root_dir=          # set root directory
dataset=FakeAVCeleb


for ((i=0;i<${#audio_noisy[@]};i++)); do
    for ((j=0;j<${#video_noisy[@]};j++)); do
        echo "audio_noisy: ${audio_noisy[$i]}, video_noisy: ${video_noisy[$j]}"

        file_path=$root_dir/data/$dataset/label_test_updated_50.csv
        infer_path=$root_dir/infer/$dataset/white_audio_${audio_noisy[$i]}_video_${video_noisy[$j]}_vsr.json
        asr_infer_path=$root_dir/infer/$dataset/white_audio_${audio_noisy[$i]}_video_${video_noisy[$j]}_asr.csv

        CUDA_VISIBLE_DEVICES=0 python predict_asr.py \
                                decode.snr_target=${audio_noisy[$i]} \
                                file_path=$file_path \
                                infer_path=$infer_path \
                                asr_infer_path=$asr_infer_path

        CUDA_VISIBLE_DEVICES=0 python predict_vsr.py \
                                decode.snr_target=${video_noisy[$j]} \
                                file_path=$asr_infer_path \
                                infer_path=$infer_path

        python roc_analyzer.py \
                --save_path=$dataset/ \
                --data_path=$dataset/white_audio_${audio_noisy[$i]}_video_${video_noisy[$j]}_vsr \
                --check_metric=wer \
                --base_dir=$root_dir
    done
done