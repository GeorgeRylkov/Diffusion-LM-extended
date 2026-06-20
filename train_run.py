import sys 
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--experiment', type=str, default='no-rep', help='no-rep=gpt2gen, no-zipfs, has-rep=regular, rm-window-rep')
    parser.add_argument('--task', type=str, default='wp', help='wp, wikitext')

    parser.add_argument('--rand_idx', type=str, default='no',
                        help='no or yes')

    parser.add_argument('--pretrained_model', type=str, default='gpt2', help='')
    parser.add_argument('--model_type', type=str, default='gpt2', help='')

    parser.add_argument('--dataset_name', type=str, default='wikitext', help='')
    parser.add_argument('--dataset_config_name', type=str, default='wikitext-103-raw-v1', help='')
    parser.add_argument('--train_file', type=str, default='wikitext', help='')
    parser.add_argument('--validation_file', type=str, default='wikitext', help='')

    parser.add_argument('--dir_name', type=str, default=None, help='')
    parser.add_argument('--notes', type=str, default=None, help='')
    parser.add_argument('--block_size', type=int, default=100, help='')

    # training parameters.
    parser.add_argument('--seed', type=int, default=101, help='') # old is 42
    parser.add_argument('--bsz', type=int, default=10, help='')
    parser.add_argument('--epoch', type=int, default=5, help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='')
    parser.add_argument('--temperature', type=float, default=1., help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')
    parser.add_argument('--percent', type=float, default=1.0, help='')

    parser.add_argument('--submit', type=str, default='no', help='')
    parser.add_argument('--use_big', type=str, default='no', help='')

    parser.add_argument('--app', type=str, default='', help='')

    # Optional overrides for run_clm / HuggingFace Trainer (classifier resume, caps, best checkpoint)
    parser.add_argument('--resume_from_checkpoint', type=str, default='',
                        help='Path to a Trainer checkpoint dir (e.g. .../checkpoint-100000). Omit --overwrite_output_dir when set.')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='If > 0, total step cap (overrides epoch-only termination).')
    parser.add_argument('--clm_save_steps', type=int, default=50000, help='run_clm --save_steps')
    parser.add_argument('--clm_eval_steps', type=int, default=10000, help='run_clm --eval_steps')
    parser.add_argument('--save_total_limit', type=int, default=1, help='HF Trainer save_total_limit')
    parser.add_argument('--load_best_model_at_end', action='store_true',
                        help='Keep best eval checkpoint (needs eval every save/eval steps)')
    parser.add_argument('--metric_for_best_model', type=str, default='eval_loss')
    parser.add_argument('--no_overwrite_output_dir', action='store_true',
                        help='Do not pass --overwrite_output_dir (required when resuming).')


    args = parser.parse_args()

    folder_name = "classifier_models"


    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    if args.experiment == 'e2e-tgt' or  args.experiment == 'e2e-tgt-pos' or args.experiment == 'e2e-tgt-tree' or \
            args.experiment == 'e2e-tgt-gen-tree' or  args.experiment == 'e2e-tgt-gen-pos' or args.experiment == 'e2e-back-gen' \
            or args.experiment == 'e2e-tgt-gen-length' or args.experiment == 'e2e-tgt-gen-spans' \
            or args.experiment == 'e2e-back' \
            or args.experiment == 'simple-wiki' or args.experiment == 'roc' or args.experiment == 'wiki':

        if args.dataset_name == 'none':
            Model_FILE = args.experiment + \
                         '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch, args.bsz * args.gradient_accumulation_steps,
                                                     args.pretrained_model, os.path.basename(args.train_file), args.seed,
                                                           args.task)
            Model_FILE = Model_FILE + f'_{args.notes}'
            logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
            Model_FILE = os.path.join(folder_name, Model_FILE)
            app = f" --train_file={args.train_file} --validation_file {args.validation_file} " \
                  f" --task {args.task}"
            app += " " + args.app


        else:
            Model_FILE = args.experiment + \
                         '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch, args.bsz * args.gradient_accumulation_steps,
                                                     args.pretrained_model, args.dataset_config_name, args.seed, args.task)
            Model_FILE = Model_FILE + f'_{args.notes}'
            logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
            Model_FILE = os.path.join(folder_name, Model_FILE)
            app = f" --dataset_name={args.dataset_name} " \
                  f"--dataset_config_name {args.dataset_config_name} --task {args.task}"
            app += " " + args.app




    overwrite = " --overwrite_output_dir " if not args.no_overwrite_output_dir else " "

    COMMANDLINE = f"python transformers/examples/pytorch/language-modeling/run_clm.py \
            --output_dir={Model_FILE} \
            --model_name_or_path={args.pretrained_model} \
            --tokenizer_name={args.pretrained_model} \
            --per_device_train_batch_size {args.bsz} \
            --per_device_eval_batch_size {args.bsz} \
            --save_steps {args.clm_save_steps} \
            --num_train_epochs {args.epoch} \
            --do_train --eval_steps {args.clm_eval_steps} --evaluation_strategy steps \
            --do_eval --dataloader_num_workers 4 \
            --save_total_limit {args.save_total_limit} \
            {overwrite} \
            --logging_dir {logging_dir} \
            --block_size {args.block_size}  \
            --disable_tqdm True --model_type {args.model_type} \
            --gradient_accumulation_steps {args.gradient_accumulation_steps} \
            --save_strategy steps \
                  --experiment {args.experiment} --seed {args.seed}"

    if args.max_steps > 0:
        COMMANDLINE += f" --max_steps {args.max_steps}"
    if args.resume_from_checkpoint:
        # Quote for shell: output_dir names may contain '='
        COMMANDLINE += f' --resume_from_checkpoint "{args.resume_from_checkpoint}"'
    if args.load_best_model_at_end:
        # HF infers greater_is_better=False when metric_for_best_model is eval_loss
        COMMANDLINE += (
            " --load_best_model_at_end "
            f"--metric_for_best_model {args.metric_for_best_model}"
        )

    COMMANDLINE += app

    with open(Model_FILE + '.sh', 'w') as f:
        print(COMMANDLINE, file=f)

    print(COMMANDLINE)
    if args.submit == 'no':
        os.system(COMMANDLINE)  # textattack/roberta-base-ag-news # textattack/roberta-base-imdb
