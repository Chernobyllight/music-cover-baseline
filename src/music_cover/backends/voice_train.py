"""Fine-tune the pinned upstream Seed-VC objective, with explicit durable resume."""
import importlib.util
import hashlib
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import yaml
from ..common.paths import ROOT
from ..common.io import write_json, identity
from ..audio import sha256
from ..voice import BASE_CHECKPOINT, BASE_CONFIG
from ..voice_weights import check_parameter_coverage


def verify_training_assets():
    lock = json.loads((ROOT / 'configs/voice_training_assets.lock.json').read_text())
    for row in lock['assets']:
        if row['scope'] != 'training': continue
        p = ROOT / row['path']
        if not p.is_file() or sha256(p) != row['sha256']:
            raise ValueError(f'Training asset missing/corrupt: {p}; run scripts/setup/setup_voice_training.py')


def local_asset(repo, filename='pytorch_model.bin', config_filename=None):
    mapping = {
        ('myshell-ai/OpenVoiceV2', 'converter/checkpoint.pth'): ROOT / 'models/OpenVoiceV2/converter/checkpoint.pth',
        ('myshell-ai/OpenVoiceV2', 'converter/config.json'): ROOT / 'models/OpenVoiceV2/converter/config.json',
        ('Plachta/Seed-VC', 'se_db.pt'): ROOT / 'models/Seed-VC/se_db.pt',
        ('funasr/campplus', 'campplus_cn_common.bin'): ROOT / 'models/CAMPPlus/campplus_cn_common.bin',
        ('lj1995/VoiceConversionWebUI', 'rmvpe.pt'): ROOT / 'models/RMVPE/rmvpe.pt',
    }
    value = str(mapping[repo, filename])
    return (value, str(mapping[repo, config_filename])) if config_filename else value


def train_voice(c, run, out, resume=False):
    import torch
    from ..voice_training import path, dataset_manifest
    verify_training_assets()
    manifest = dataset_manifest(c)
    t = c['training']
    seed = int(t.get('seed', 42))
    checkpoint = path(c.get('base_checkpoint', BASE_CHECKPOINT))
    source_config = path(c.get('base_config', BASE_CONFIG))
    # Validation/test/prompt folders never enter the upstream recursive dataset reader.
    train_dir = path(c['dataset_dir']) / 'train'
    cfg = yaml.safe_load(source_config.read_text())
    cfg['log_dir'] = str(run / 'upstream')
    cfg['pretrained_model'] = ''
    cfg['model_params']['vocoder']['name'] = str(ROOT / 'models/BigVGAN')
    cfg['model_params']['speech_tokenizer']['name'] = str(ROOT / 'models/Whisper-small')
    model_config = run / 'model_config.yml'
    config_text = yaml.safe_dump(cfg, sort_keys=False)
    stable = {'dataset': manifest['identity'], 'base_sha256': sha256(checkpoint),
              'model_config_sha256': hashlib.sha256(config_text.encode()).hexdigest(),
              'training': {k: v for k, v in t.items() if k not in {'max_steps', 'save_every'}},
              'upstream_sha256': sha256(ROOT / 'third_party/seed-vc/train.py')}
    key = identity(stable)
    ckpt_dir = run / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    latest = ckpt_dir / 'latest.pth'
    if latest.exists() and not resume: raise ValueError('Checkpoint exists; use --resume or a new training --run')
    if resume and not latest.is_file(): raise ValueError('--resume requested but latest.pth is absent')
    state = torch.load(latest, map_location='cpu', weights_only=False) if resume else None
    if state is not None and state['training_identity'] != key:
        raise ValueError('Resume dataset/base/config/hyperparameters changed')
    model_config.write_text(config_text)
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    upstream_dir = ROOT / 'third_party/seed-vc'
    os.chdir(upstream_dir)
    sys.path.insert(0, str(upstream_dir))
    spec = importlib.util.spec_from_file_location('seed_vc_training_upstream', upstream_dir / 'train.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.load_custom_model_from_hf = local_asset
    base_loader = upstream.load_checkpoint
    def strict_parameters(model, optimizer, filename, **kwargs):
        weights = torch.load(filename, map_location='cpu', weights_only=False)['net']
        check_parameter_coverage(model, weights)
        del weights
        return base_loader(model, optimizer, filename, **kwargs)
    upstream.load_checkpoint = strict_parameters
    trainer = upstream.Trainer(str(model_config), str(checkpoint), str(train_dir), 'fit',
        batch_size=t['batch_size'], num_workers=0, steps=t['max_steps'], save_interval=t['save_every'], device='cuda:0')
    for name in ('whisper_model', 'campplus_model', 'bigvgan_model'):
        model = getattr(trainer, name, None)
        if model is not None: model.eval().requires_grad_(False)
    trainer.tone_color_converter.model.eval().requires_grad_(False)
    if hasattr(trainer.rmvpe, 'model'): trainer.rmvpe.model.eval().requires_grad_(False)
    for key_name, optim in trainer.optimizer.optimizers.items():
        for group in optim.param_groups: group['lr'] = t.get('learning_rate', 1e-5)
        trainer.optimizer.schedulers[key_name].base_lrs = [t.get('learning_rate', 1e-5)] * len(optim.param_groups)
    step = 0
    if resume:
        for name, model in trainer.model.items(): model.load_state_dict(state['net'][name], strict=True)
        for name, values in state['optimizer']: trainer.optimizer.optimizers[name].load_state_dict(values)
        for name, values in state['scheduler']: trainer.optimizer.schedulers[name].load_state_dict(values)
        torch.set_rng_state(state['rng']['torch'])
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
        random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
        step = int(state['iters'])
        del state
    if step >= t['max_steps']: raise ValueError('max_steps must exceed the restored step')
    started = step
    snapshots = {k: [p.detach().flatten()[:128].cpu().clone() for p in net.parameters()]
                 for k, net in trainer.model.items()}
    for net in trainer.model.values(): net.train()
    from data.ft_dataset import collate
    dataset = trainer.train_dataloader.dataset
    log_path = run / 'loss.jsonl'
    last_loss = None
    def save():
        state = {'net': {k: net.state_dict() for k, net in trainer.model.items()},
                 'optimizer': trainer.optimizer.state_dict(), 'scheduler': trainer.optimizer.scheduler_state_dict(),
                 'iters': step, 'epoch': 0, 'training_identity': key,
                 'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                         'python': random.getstate(), 'numpy': np.random.get_state()}}
        target = ckpt_dir / f'step_{step:06d}.pth'
        temp = target.with_suffix('.tmp')
        torch.save(state, temp); temp.replace(target)
        link = ckpt_dir / 'latest.tmp'
        link.unlink(missing_ok=True); link.symlink_to(target.name); link.replace(latest)
        return target
    while step < t['max_steps']:
        # Stateless per-step sampling makes resume independent of DataLoader iterator position.
        sampler = random.Random(seed + step)
        batch = collate([dataset[sampler.randrange(len(dataset))] for _ in range(t['batch_size'])])
        last_loss = trainer.train_one_step([x.to('cuda:0') for x in batch])
        if not np.isfinite(last_loss): raise RuntimeError('Non-finite training loss; no model exported')
        step += 1
        with log_path.open('a') as stream:
            stream.write(json.dumps({'step': step, 'loss': last_loss}) + '\n')
        if step % 10 == 0 or step == started + 1: print(f'train step={step} loss={last_loss:.6f}', flush=True)
        if step % t['save_every'] == 0 or step == t['max_steps']: final = save()
    changes = {}
    for name, net in trainer.model.items():
        changes[name] = sum(not torch.equal(old, p.detach().flatten()[:128].cpu())
                            for old, p in zip(snapshots[name], net.parameters()))
    write_json(out / 'training.json', {'training_identity': key, 'inputs': stable,
        'start_step': started, 'end_step': step, 'last_loss': last_loss, 'checkpoint': str(final),
        'checkpoint_sha256': sha256(final), 'resumed_optimizer_scheduler_rng': resume,
        'changed_parameter_tensor_samples': changes,
        'trainable_parameters': {k: sum(p.numel() for p in net.parameters() if p.requires_grad) for k, net in trainer.model.items()},
        'frozen_helpers': ['Whisper', 'CAMPPlus', 'RMVPE', 'OpenVoiceV2', 'BigVGAN'],
        'quality_status': 'not_evaluated', 'smoke_test': bool(c.get('smoke_test'))})
