"""Analyze token ID patterns in generated samples to check # vs ## distribution."""
import sys, json, torch, numpy as np
from collections import Counter
sys.path.insert(0, '.')
from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from transformers import AutoTokenizer

model_dir = sys.argv[1]
npz_path = sys.argv[2]

t = AutoTokenizer.from_pretrained(model_dir)

with open(f'{model_dir}/training_args.json', 'r') as f:
    ta = json.load(f)
defaults = model_and_diffusion_defaults()
defaults.update(ta)
model, diffusion = create_model_and_diffusion(**defaults)
state = torch.load(f'{model_dir}/ema_0.9999_400000.pt', map_location='cpu')
model.load_state_dict(state)
model.eval().cuda()

npz = np.load(npz_path)
arr = npz['arr_0'][:50]
print(f'Analyzing {len(arr)} samples...\n')

x_t = torch.tensor(arr).float().cuda()
with torch.no_grad():
    logits = model.get_logits(x_t)
    all_ids = logits.argmax(dim=-1).cpu()

total_hash = 0
total_subword = 0
hash_run_lengths = []
samples_with_hash = 0

for idx, sample_ids in enumerate(all_ids):
    tokens = t.convert_ids_to_tokens(sample_ids.tolist())
    ids = sample_ids.tolist()
    run = 0
    sample_has_hash = False
    for tok in tokens:
        if tok == '#':
            run += 1
            total_hash += 1
            sample_has_hash = True
        else:
            if run > 0:
                hash_run_lengths.append(run)
            run = 0
            if tok.startswith('##'):
                total_subword += 1
    if run > 0:
        hash_run_lengths.append(run)
    if sample_has_hash:
        samples_with_hash += 1

    if idx < 5:
        print(f'--- Sample {idx} ---')
        print(f'IDs:    {ids}')
        print(f'Tokens: {tokens}')
        print(f'Decode: {t.decode(sample_ids.tolist())}')
        print()

print(f'=== Stats across {len(arr)} samples ===')
print(f'Samples containing #: {samples_with_hash}/{len(arr)}')
print(f'Total standalone # (id=108): {total_hash}')
print(f'Total ## subword tokens: {total_subword}')
print(f'Avg # per sample: {total_hash/len(arr):.1f}')
print(f'Consecutive # run lengths: {Counter(hash_run_lengths)}')

if hash_run_lengths:
    always_two = all(r == 2 for r in hash_run_lengths)
    print(f'Are ALL # runs exactly length 2? {always_two}')
