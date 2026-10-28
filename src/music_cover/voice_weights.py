"""Validate every trainable tensor using the upstream DDP key normalization."""
def check_parameter_coverage(model, state):
    missing = []
    for key, network in model.items():
        weights = {name.removeprefix('module.'): value for name, value in state.get(key, {}).items()}
        for name, parameter in network.named_parameters():
            if name not in weights or weights[name].shape != parameter.shape:
                missing.append(f'{key}.{name}')
    if missing:
        raise ValueError(f'Incomplete/incompatible voice weights: {missing[:20]}')
