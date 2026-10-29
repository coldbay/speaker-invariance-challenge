"""Create/recover the local MOSS venv, including Debian without ensurepip.

Fallback uses the official PyPA standalone virtualenv zipapp, never apt/sudo
or an installation into the original Python environment. Existing incomplete
venvs are renamed within the project; existing healthy environments are reused.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def environment_python(environment):
    return environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def healthy(environment):
    executable = environment_python(environment)
    if not executable.is_file():
        return False
    check = ('import sys,pip; from pathlib import Path; '
             'assert Path(pip.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())')
    return subprocess.run([str(executable), '-I', '-c', check],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def preserve_incomplete(environment):
    if environment.is_symlink():
        raise ValueError('Refusing a symlink environment directory.')
    if not environment.exists():
        return
    # A move is allowed only for the named environment directly under ROOT.
    if environment.is_symlink() or environment.parent.resolve() != ROOT:
        raise ValueError('Refusing to move an environment outside the project.')
    if not (environment / 'pyvenv.cfg').is_file():
        raise ValueError(f'Existing directory is not a recognizable venv: {environment}')
    backup = environment.with_name(environment.name + '.incomplete.' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if backup.parent.resolve() != ROOT:
        raise ValueError('Backup must remain inside the project.')
    environment.rename(backup)
    print('Preserved incomplete environment:', backup.name, flush=True)


def bootstrap_virtualenv(environment):
    bootstrap = ROOT / '.moss-bootstrap'
    bootstrap.mkdir(exist_ok=True)
    tag = f'{sys.version_info.major}.{sys.version_info.minor}'
    path = bootstrap / f'virtualenv-{tag}.pyz'
    url = f'https://bootstrap.pypa.io/virtualenv/{tag}/virtualenv.pyz'
    if not path.exists():
        print('Downloading official virtualenv bootstrap:', url, flush=True)
        temporary = path.with_suffix('.part')
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    version = subprocess.check_output([sys.executable, str(path), '--version'], text=True).strip()
    (bootstrap / f'virtualenv-{tag}.manifest.json').write_text(
        json.dumps({'url': url, 'sha256': digest, 'version': version}, indent=2) + '\n', encoding='utf-8')
    subprocess.run([sys.executable, str(path), '--system-site-packages', '--no-download',
                    '--no-periodic-update', '--app-data', str(bootstrap / 'cache'), str(environment)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='.venv-moss-reuse')
    parser.add_argument('--creator', choices=('auto', 'virtualenv'), default='auto')
    args = parser.parse_args()
    if Path(args.name).name != args.name or not args.name.startswith('.venv-moss-'):
        raise ValueError('Use a single .venv-moss-* directory name inside the project.')
    environment = ROOT / args.name
    if environment.is_symlink():
        raise ValueError('Refusing a symlink environment directory.')
    if healthy(environment):
        print('Reusing environment:', environment.name, flush=True)
        return
    preserve_incomplete(environment)
    if args.creator == 'auto' and importlib.util.find_spec('ensurepip') is not None:
        created = subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(environment)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if created.returncode == 0 and healthy(environment):
            print('Created environment with venv:', environment.name, flush=True)
            return
        print('Standard venv could not bootstrap pip; switching to virtualenv.', flush=True)
        preserve_incomplete(environment)
    bootstrap_virtualenv(environment)
    if not healthy(environment):
        raise RuntimeError('Environment still lacks its own working pip.')
    print('Environment ready:', environment.name, flush=True)


if __name__ == '__main__':
    main()
