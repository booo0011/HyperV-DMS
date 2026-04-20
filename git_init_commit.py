import os
import subprocess
from pathlib import Path

repo = Path('e:/项目/超图项目/HyperV-DMS')
git = Path('E:/Git/cmd/git.exe')
print('git exists', git.exists())
print('repo exists', repo.exists())
print('.git exists', (repo / '.git').exists())
os.chdir(repo)

if not (repo / '.git').exists():
    r = subprocess.run([str(git), 'init'], capture_output=True, text=True, errors='replace')
    print('init rc', r.returncode)
    print('init out', r.stdout)
    print('init err', r.stderr)

r = subprocess.run([str(git), 'status', '--short', '--branch'], capture_output=True, text=True, errors='replace')
print('status rc', r.returncode)
print(r.stdout)
print(r.stderr)

lock_path = repo / '.git' / 'index.lock'
if lock_path.exists():
    lock_path.unlink()
    print('removed stale lock', lock_path)

r = subprocess.run([str(git), 'add', '.'], capture_output=True, text=True, errors='replace')
print('add rc', r.returncode)
print('add err', r.stderr[:2000])

r = subprocess.run([str(git), 'commit', '-m', 'Initial commit'], capture_output=True, text=True, errors='replace')
print('commit rc', r.returncode)
print('commit out', r.stdout[:2000])
print('commit err', r.stderr[:2000])

r = subprocess.run([str(git), 'branch', '-M', 'main'], capture_output=True, text=True, errors='replace')
print('branch rc', r.returncode)
print('branch err', r.stderr[:2000])

# set origin remote, remove if exists
subprocess.run([str(git), 'remote', 'remove', 'origin'], capture_output=True, text=True, errors='replace')
remote_add = subprocess.run([str(git), 'remote', 'add', 'origin', 'https://github.com/booo0011/HyperV-DMS.git'], capture_output=True, text=True, errors='replace')
print('remote add rc', remote_add.returncode)
print('remote add err', remote_add.stderr[:2000])

r = subprocess.run([str(git), 'remote', '-v'], capture_output=True, text=True, errors='replace')
print('remotes rc', r.returncode)
print(r.stdout)
print(r.stderr)
