import os, glob, json, argparse, random

parser = argparse.ArgumentParser()
parser.add_argument('model_glob', type=str, help='Glob pattern for model directories')
parser.add_argument('--top_p', type=float, default=-1.0)
parser.add_argument('--pattern', type=str, default='model', help='Checkpoint filename pattern')
parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to generate')
parser.add_argument('--gen_refs', action='store_true',
                    help='After decode/PPL, sample reference lines from --ref_file for downstream MAUVE')
parser.add_argument('--ref_file', type=str, default='datasets/ROCstory/roc_valid.json',
                    help='Path to reference dataset (JSON, one story per line)')
parser.add_argument('--ref_out', type=str, default=None,
                    help='Where to write --gen_refs output (default: <out_dir>/roc_references.txt)')
parser.add_argument('--n_refs', type=int, default=1000,
                    help='Number of reference texts to sample')
parser.add_argument('--ref_seed', type=int, default=42, help='Seed for reference sampling')
parser.add_argument('--out_dir', type=str, default='generation_outputs')
bd_args = parser.parse_args()

full_lst = glob.glob(bd_args.model_glob)
top_p = bd_args.top_p
print(f'top_p = {top_p}')
pattern_ = bd_args.pattern
print(f'pattern_ = {pattern_}')

output_lst = []
for lst in full_lst:
    print(lst)
    try:
        tgt = sorted(glob.glob(f"{lst}/{pattern_}*pt"))[-1]
        lst = os.path.split(lst)[1]
        print(lst)
        num = 1
    except:
        continue
    model_arch_ = lst.split('_')[5-num]
    model_arch = 'conv-unet' if 'conv-unet' in lst else 'transformer'
    mode =  'image' if ('conv' in model_arch ) else 'text' #or '1d-unet' in model_arch_
    print(mode, model_arch_)
    dim_ =lst.split('_')[4-num]

    # diffusion_steps= 4000
    # noise_schedule = 'cosine'
    # dim = dim_.split('rand')[1]

    if 'synth' in lst:
        modality = 'synth'
    elif 'pos' in lst:
        modality = 'pos'
    elif 'image' in lst:
        modality = 'image'
    elif 'roc' in lst:
        modality = 'roc'
    elif 'simple-wiki' in lst:
        modality = 'simple-wiki'
    elif 'wiki' in lst:
        modality = 'wiki'
    elif 'e2e-tgt' in lst:
        modality = 'e2e-tgt'
    elif 'book' in lst:
        modality = 'book'
    elif 'yelp' in lst:
        modality = 'yelp'
    elif 'commonGen' in lst:
        modality = 'commonGen'
    elif 'e2e' in lst:
        modality = 'e2e'


    if 'synth32' in lst:
        kk = 32
    elif 'synth128' in lst:
        kk = 128

    try:
        diffusion_steps = int(lst.split('_')[7-num])
        print(diffusion_steps)
    except:
        diffusion_steps = 4000
    try:
        noise_schedule = lst.split('_')[8-num]
        assert  noise_schedule in ['cosine', 'linear']
        print(noise_schedule)
    except:
        noise_schedule = 'cosine'
    try:
        dim = int(dim_.split('rand')[1])
    except:
        dim =lst.split('_')[4-num]
    try:
        print(len(lst.split('_')))
        num_channels =  int(lst.split('_')[-1].split('h')[1])
    except:
        num_channels = 128

    print(tgt, model_arch, dim, num_channels)
    # out_dir = 'diffusion_lm/improved_diffusion/out_gen_large_nucleus'
    # num_samples = 512

    # out_dir = 'diffusion_lm/improved_diffusion/out_gen_v2_nucleus'

    out_dir = bd_args.out_dir
    num_samples = bd_args.num_samples

    if modality == 'e2e':
        num_samples = max(num_samples, 547)

    COMMAND = f'python scripts/{mode}_sample.py ' \
    f'--model_path {tgt} --batch_size 50 --num_samples {num_samples} --top_p {top_p} ' \
    f'--out_dir {out_dir} '
    print(COMMAND)
    # os.system(COMMAND)

    # shape_str = "x".join([str(x) for x in arr.shape])
    model_base_name = os.path.basename(os.path.split(tgt)[0]) + f'.{os.path.split(tgt)[1]}'
    if modality == 'e2e-tgt' or modality == 'e2e':
        out_path2 = os.path.join(out_dir, f"{model_base_name}.samples_{top_p}.json")
    else:
        out_path2 =  os.path.join(out_dir, f"{model_base_name}.samples_{top_p}.txt")
    output_cands = glob.glob(out_path2)
    print(out_path2, output_cands)
    if len(output_cands) > 0:
        out_path2 = output_cands[0]
    else:
        ret = os.system(COMMAND)
        if ret != 0:
            print(f"ERROR: sampling command failed with exit code {ret}, skipping {lst}")
            continue
        output_cands = glob.glob(out_path2)
        if len(output_cands) == 0:
            print(f"ERROR: no output file found at {out_path2} after sampling, skipping {lst}")
            continue
        out_path2 = output_cands[0]

    output_lst.append(out_path2)

    if modality == 'pos':
        model_name_path = 'predictability/diff_models/pos_e=15_b=20_m=gpt2_wikitext-103-raw-v1_s=102'
    elif modality == 'synth':
        if kk == 128:
            model_name_path = 'predictability/diff_models/synth_e=15_b=10_m=gpt2_wikitext-103-raw-v1_None'
        else:
            model_name_path = 'predictability/diff_models/synth_e=15_b=20_m=gpt2_wikitext-103-raw-v1_None'
    elif modality == 'e2e-tgt':
        if 'bert_uncased' in lst or 'bert_tiny_frozen' in lst:
            model_name_path = "../classifier_models/e2e-tgt_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_bert_uncased"
        else:
            model_name_path = "../classifier_models/e2e-tgt_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_None"
    elif modality == 'roc':
        if 'bert_uncased' in lst or 'bert_tiny_frozen' in lst:
            model_name_path = "../classifier_models/roc_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_bert_uncased"
        elif 'bert' in lst or 'rand768' in lst:
            model_name_path = "../classifier_models/roc_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_bert_v2"
        elif 'gpt2' in lst:
            model_name_path = "../classifier_models/roc_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_gpt2"
        elif 'fasttext' in lst:
            model_name_path = "../classifier_models/roc_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_fasttext"
        else:
            model_name_path = "../classifier_models/roc_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_None"
    elif modality == 'wiki':
        if 'bert_uncased' in lst or 'bert_tiny_frozen' in lst:
            # Single trained classifier for now (partial wiki); use for full-wiki diffusion PPL too.
            # bert_tiny_frozen runs use bert-base-uncased tokenizer -> same AR head as bert_uncased.
            model_name_path = (
                "../classifier_models/"
                "wiki_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_bert_uncased_wiki_partial"
            )
        else:
            model_name_path = "../classifier_models/wiki_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_None"
    elif modality == 'e2e':
        COMMAND1 = f"python diffusion_lm/e2e_data/mbr.py {out_path2}"

        os.system(COMMAND1)
        COMMAND2 = f"python e2e-metrics/measure_scores.py " \
                   f"diffusion_lm/improved_diffusion/out_gen_v2_dropout2/1_valid_gold  " \
                   f"{out_path2}.clean -p  -t -H > {os.path.join(os.path.split(tgt)[0], 'e2e_valid_eval.txt')}"
        print(COMMAND2)
        os.system(COMMAND2)
        continue
    else:
        print('not trained a AR model yet... only look at the output plz.')
        continue
    if 'bert' in lst and 'rand' not in dim_:
        experiment_type = 'bert'
    else:
        experiment_type = 'random'
    COMMAND = f"python scripts/ppl_under_ar.py " \
              f"--model_path {tgt} " \
              f"--modality {modality}  --experiment {experiment_type} " \
              f"--model_name_or_path {model_name_path} " \
              f"--input_text {out_path2}  --mode eval"

    print(COMMAND)
    print()
    os.system(COMMAND)
print('output lists:')
print("\n".join(output_lst))

if bd_args.gen_refs:
    random.seed(bd_args.ref_seed)
    ref_path = bd_args.ref_file
    print(f'\nGenerating {bd_args.n_refs} reference texts from {ref_path}')
    stories = []
    with open(ref_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                stories.append(json.loads(line)[0])
    sampled = random.sample(stories, min(bd_args.n_refs, len(stories)))
    ref_out = bd_args.ref_out or os.path.join(bd_args.out_dir, 'roc_references.txt')
    with open(ref_out, 'w', encoding='utf-8') as f:
        for s in sampled:
            # One record per line for MAUVE (load_texts reads line-by-line).
            # Wiki passages often contain paragraph breaks (\n) inside the JSON string.
            f.write(' '.join(s.split()) + '\n')
    print(f'Written {len(sampled)} reference texts to {ref_out}')