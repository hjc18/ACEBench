python generate.py --model Qwen3-8B-local \
    --model-path /cloud/oss_checkpoints/Qwen3/Qwen3-8B \
    --category normal --language en

    # --model-path /storage/STAR/scripts/ckd/tmp_checkpoints/global_step1100_hf \
python eval_main.py --model Qwen3-8B-local --category normal --language en