python train.py --data_dir ./data/DocRED \
--transformer_type bert \
--model_name_or_path ./pretrain_models/bert-base-cased \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed 66 \
--num_class 97 \
--save_path ./saved_model/run_bert_pig.pth


# python train.py --data_dir ./data/DocRED \
# --transformer_type bert \
# --model_name_or_path ./pretrain_models/bert-base-cased \
# --train_file train_annotated.json \
# --dev_file dev.json \
# --test_file test.json \
# --train_batch_size 16 \
# --test_batch_size 32 \
# --gradient_accumulation_steps 1 \
# --num_labels 4 \
# --learning_rate 5e-8 \
# --max_grad_norm 1.0 \
# --warmup_ratio 0.06 \
# --num_train_epochs 30.0 \
# --seed 66 \
# --num_class 97 \
# --save_path ./saved_model/run_bert.pth


## predict
# python train.py --data_dir ./data/DocRED \
# --transformer_type bert \
# --model_name_or_path ./pretrain_models/bert-base-cased \
# --train_file dev_only_evidence.json \
# --dev_file dev.json \
# --test_file test.json \
# --train_batch_size 16 \
# --test_batch_size 64 \
# --gradient_accumulation_steps 1 \
# --num_labels 4 \
# --learning_rate 5e-5 \
# --max_grad_norm 1.0 \
# --warmup_ratio 0.06 \
# --num_train_epochs 30.0 \
# --seed 66 \
# --num_class 97 \
# --load_path ./saved_model/run_bert.pth