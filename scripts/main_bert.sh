CUDA_VISIBLE_DEVICES=1 python main_branch/train.py --data_dir ./dataset/docred \
--transformer_type bert \
--model_name_or_path ./pretrain_models/bert-base-cased \
--train_file train_annotated.json \
--dev_file train_annotated.json \
--test_file test.json \
--train_batch_size 4 \
--test_batch_size 4 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--max_seq_length 512 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 40.0 \
--seed 66 \
--num_class 97 \
--desc run-original-atlop-with-selen-512 \
--gpu 0 \
--save_path ./saved_model/main_branch/baseline_atlop_512seqlen.pth


# CUDA_VISIBLE_DEVICES=1 python main_branch/train.py --data_dir ./dataset/docred \
# --transformer_type bert \
# --model_name_or_path ./pretrain_models/bert-base-cased \
# --train_file train_annotated.json \
# --dev_file dev.json \
# --test_file test.json \
# --train_batch_size 4 \
# --test_batch_size 4 \
# --gradient_accumulation_steps 1 \
# --num_labels 4 \
# --max_seq_length 500 \
# --learning_rate 5e-5 \
# --max_grad_norm 1.0 \
# --warmup_ratio 0.06 \
# --num_train_epochs 40.0 \
# --seed 66 \
# --num_class 97 \
# --desc run-original-atlop-with-selen-500 \
# --gpu 0 \
# --save_path ./saved_model/main_branch/baseline_atlop_500seqlen.pth


# CUDA_VISIBLE_DEVICES=1 python main_branch/train.py --data_dir ./dataset/docred \
# --transformer_type bert \
# --model_name_or_path ./pretrain_models/bert-base-cased \
# --train_file train_annotated.json \
# --dev_file dev.json \
# --test_file test.json \
# --train_batch_size 4 \
# --test_batch_size 4 \
# --gradient_accumulation_steps 1 \
# --num_labels 4 \
# --max_seq_length 500 \
# --learning_rate 5e-5 \
# --max_grad_norm 2.0 \
# --warmup_ratio 0.05 \
# --num_train_epochs 40.0 \
# --seed 66 \
# --num_class 97 \
# --desc run-original-atlop-with-selen-500 \
# --gpu 0 \
# --save_path ./saved_model/main_branch/baseline_atlop_500seqlen.pth



# CUDA_VISIBLE_DEVICES=0 python main_branch/train.py --data_dir ./dataset/docred \
# --transformer_type bert \
# --model_name_or_path ./pretrain_models/bert-base-cased \
# --train_file train_annotated.json \
# --dev_file dev.json \
# --test_file test.json \
# --train_batch_size 4 \
# --test_batch_size 4 \
# --gradient_accumulation_steps 1 \
# --num_labels 4 \
# --learning_rate 5e-5 \
# --max_grad_norm 1.0 \
# --warmup_ratio 0.06 \
# --num_train_epochs 30.0 \
# --seed 66 \
# --num_class 97 \
# --desc run-original-atlop-only-in-class-with-f1-below-0.5-with-more-negs-using-balancesloss-docunet \
# --save_path ./saved_model/main_branch/atlop_in_tail_classv3.pth


## predict
# python train.py --data_dir ./dataset/docred \
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