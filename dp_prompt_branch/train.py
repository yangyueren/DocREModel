import argparse
import os
import sys
sys.path.append('.')

from dp_prompt_branch.prepro import Verbalizer


import numpy as np
import torch
# from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig,  AutoTokenizer, AutoModelForMaskedLM
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler


from dp_prompt_branch.model import DocREModel
from dp_prompt_branch.utils import set_seed, collate_fn
from dp_prompt_branch.prepro import read_docred
from dp_prompt_branch.evaluation import to_official, official_evaluate, official_evaluate_prompt
import wandb
from tqdm import tqdm

from dp_prompt_branch.mylogger import logger

def finetune(args, features, dev_features, model, optimizer, num_epoch, num_steps, checkpoint=None):
    best_score = -1
    train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=0)
    train_iterator = range(int(num_epoch))
    total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    if checkpoint is not None:
        train_iterator = range(checkpoint['epoch'], int(num_epoch))
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info('load scheduler state dict success.')
    print("Total steps: {}".format(total_steps))
    print("Warmup steps: {}".format(warmup_steps))

    scaler = GradScaler()
    model = torch.nn.DataParallel(model)
    for epoch in train_iterator:
        model.zero_grad()
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader),desc="pig"):
            inputs = {'input_ids': batch[0].to(args.device),
                        'attention_mask': batch[1].to(args.device),
                        'labels': batch[2],
                        'entity_pos': batch[3],
                        'hts': batch[4],
                        }
            with autocast():
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                # yyybug
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()

                
                scheduler.step()

                optimizer.zero_grad()
                model.zero_grad()
                num_steps += 1

            wandb.log({"loss": loss.item()}, step=num_steps)
            logger.debug({"loss": loss.item()})

            # if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
        
        if epoch % args.evaluation_epoch == 0:
            dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
            wandb.log(dev_output, step=num_steps)
            logger.info(dev_output)
            if dev_score > best_score:
                best_score = dev_score
                # yyybug disable predict on test
                # pred = report(args, model, test_features) 
                # with open("result.json", "w") as fh:
                #     json.dump(pred, fh)
                if args.save_path != "":
                    torch.save({'model_state_dict':model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                # 'amp': amp.state_dict(),
                                'epoch': epoch,
                                'step': num_steps,
                                }, 
                                args.save_path)
    return num_steps
    
def train(args, model, train_features, dev_features, test_features):
    

    # freeze 11 layers.
    for name ,param in model.named_parameters():
        param.requires_grad = True
        # if 'bert.encoder.layer' in name:
        #     if '11' not in name:
        #         param.requires_grad = False
    
    new_layer = ["extractor", "bilinear"]
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    # ]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer) and p.requires_grad], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer) and p.requires_grad], "lr": 1e-4},
    ]

    

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    
    checkpoint = None
    if args.checkpoint and args.load_path != '':
        logger.info(f'Loading checkpoint ...')
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        num_steps = checkpoint['step'] if 'step' in checkpoint else 22889

        args.warmup_ratio = 0.0
        logger.info(f'Continue training from epoch: {epoch}, num_steps: {num_steps}')
    

    
    model.zero_grad()
    finetune(args, train_features, dev_features, model, optimizer, args.num_train_epochs, num_steps, checkpoint)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=0)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    # with open('./dataset/docred_pred/dev_pred_result_atlop_in_tail_classv3.json', 'w') as f:
    #     json.dump(ans, f)

    best_f1, best_f1_ign = 0, 0
    re_p, re_r = 0, 0
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _, re_p, re_r = official_evaluate_prompt(ans, args.data_dir, args.dev_file, args.sample_ratio)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_Precision": re_p * 100,
        tag + "_Recall": re_r * 100,
    }
    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=0)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--checkpoint", action='store_true')

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_epoch", default=1, type=int,
                        help="Number of training epoches between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    
    
    parser.add_argument("--gpu", default=0, type=int,
                        help="GPU for training.")
    parser.add_argument("--desc", default="see codes.", type=str)
    parser.add_argument("--predict_type", default="prompt", type=str)
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="Number of sample to use for debug.")

    args = parser.parse_args()
    logger.info(args)
    wandb.init(project="DocRED")
    # wandb.run.log_code(".")
    # backup codes.
    os.system(f"mkdir {wandb.run.dir}/codes; cp -r dp_prompt_branch/ {wandb.run.dir}/codes/")
    logger.info(f'run cp -r dp_prompt_branch/ {wandb.run.dir}/codes/ to copy python files to {wandb.run.dir}/codes')

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    docred_rel2id = json.load(open('./dataset/docred/DocRED_baseline_metadata/rel2id.json', 'r'))
    config.verbalizer = Verbalizer(docred_rel2id, tokenizer)

    logger.info(f'will sample data with ratio: {args.sample_ratio}')

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length, sample_ratio=args.sample_ratio)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length, sample_ratio=args.sample_ratio)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length, sample_ratio=args.sample_ratio)

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    model.resize_token_embeddings(len(tokenizer))

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.mask_token_id = tokenizer.mask_token_id
    config.transformer_type = args.transformer_type
    config.predict_type = args.predict_type

    logger.info('-'*40)
    logger.info(f'predict_type {args.predict_type}')
    logger.info(f'sample ratio {args.sample_ratio}')


    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(device)

    if args.load_path == "" or args.checkpoint:  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        logger.info("test...")
        # import pdb; pdb.set_trace()
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
    
        model.load_state_dict(torch.load(args.load_path)['model_state_dict'])

        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_score, dev_output)
        logger.info(dev_score, dev_output)
        # pred = report(args, model, test_features)
        # with open("result.json", "w") as fh:
        #     json.dump(pred, fh)


if __name__ == "__main__":
    main()
