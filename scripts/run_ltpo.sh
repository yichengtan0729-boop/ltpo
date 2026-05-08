dataset="Maxwell-Jia/AIME_2024"
model="Qwen/Qwen2.5-7B-Instruct"
max_new_tokens=1024
max_num_steps=10
num_thought_tokens=8
sigma=4
sigma_decay=0.9
lr=0.04
verbose=1

python main.py \
    --method ltpo \
    --dataset $dataset \
    --model_name_or_path $model \
    --output_dir ./output \
    --device cuda \
    --max_new_tokens $max_new_tokens \
    --max_num_steps $max_num_steps \
    --num_thought_tokens $num_thought_tokens \
    --sigma $sigma \
    --sigma_decay $sigma_decay \
    --lr $lr \
    --verbose $verbose
