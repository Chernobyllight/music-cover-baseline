"""Environment names shared by macOS installation, assembly and execution."""

ENVIRONMENTS = {
    'control': 'control',
    'sheetsage2': 'sheetsage2',
    'yue2': 'yue2-torch215',
    'separator': 'separator-torch215',
    'seed-vc': 'seed-vc-torch215',
}
WORKSPACE = 'envs/mps-optimized'


def lock_path(root, name):
    return root / 'configs/macos' / f'{ENVIRONMENTS[name]}.txt'
