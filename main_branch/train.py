import argparse
import os
import sys
sys.path.append('.')

import numpy as np
import torch
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from main_branch.model import DocREModel
from main_branch.utils import set_seed, collate_fn
from main_branch.prepro import read_docred
from main_branch.evaluation import to_official, official_evaluate
import wandb
from tqdm import tqdm

from common.mylogger import logger


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps, checkpoint=None):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        if checkpoint is not None:
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info('load scheduler state dict success.')
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))


        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in tqdm(enumerate(train_dataloader),desc="pig"):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          'candidates_rel': batch[5]
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    model.zero_grad()
                    num_steps += 1
                wandb.log({"loss": loss.item()}, step=num_steps)
                logger.debug({"loss": loss.item()})
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
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
                                        'amp': amp.state_dict(),
                                        'epoch': epoch,
                                        'step': num_steps,
                                        }, 
                                        args.save_path)
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    
    checkpoint = None
    if args.checkpoint and args.load_path != '':
        logger.info(f'Loading checkpoint ...')
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        num_steps = checkpoint['step'] if 'step' in checkpoint else 22889
        if 'amp' in checkpoint:
            amp.load_state_dict(checkpoint['amp'])
        args.num_train_epochs = (int(args.num_train_epochs) - int(epoch))
        args.warmup_ratio = 0.0
        logger.info(f'Continue training from epoch: {epoch}, num_steps: {num_steps}')
    
    
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps, checkpoint)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'candidates_rel': batch[5]
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    with open('./dataset/docred_pred/dev_pred_result_atlop_in_tail_classv2.json', 'w') as f:
        json.dump(ans, f)

    best_f1, best_f1_ign = 0, 0
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'candidates_rel': batch[5]
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
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    
    parser.add_argument("--gpu", default=0, type=int,
                        help="GPU for training.")
    parser.add_argument("--desc", default="see codes.", type=str)

    args = parser.parse_args()
    logger.info(args)
    wandb.init(project="DocRED")
    # wandb.run.log_code(".")
    # backup codes.
    os.system(f"mkdir {wandb.run.dir}/codes; cp -r main_branch/ {wandb.run.dir}/codes/")
    logger.info(f'run cp -r main_branch/ {wandb.run.dir}/codes/ to copy python files to {wandb.run.dir}/codes')

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

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(device)

    if args.load_path == "" or args.checkpoint:  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        logger.info("test...")
        # import pdb; pdb.set_trace()
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        ck = torch.load(args.load_path)

        if 'model_state_dict' in ck:
            model.load_state_dict(torch.load(args.load_path)['model_state_dict'])
        else:
            model.load_state_dict(ck)
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_score, dev_output)
        logger.info(dev_score, dev_output)
        # pred = report(args, model, test_features)
        # with open("result.json", "w") as fh:
        #     json.dump(pred, fh)


if __name__ == "__main__":
    main()
