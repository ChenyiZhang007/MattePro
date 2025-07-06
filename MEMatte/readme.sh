# train
CUDA_VISIBLE_DEVICES=2,3 python main.py \
    --config-file configs/Ab_ViTMatte_S_topk0.75.py \
    --num-gpus 2 \
    --dist-url tcp://127.0.0.1:14558

# inference
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/ \
    --data-dir /opt/data/private/lyh/Datasets/UHRIM/Test \
    --max-number-token 16000 \
    --patch-decoder
