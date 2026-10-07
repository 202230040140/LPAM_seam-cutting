import tempfile
from pathlib import Path
import subprocess
import sys
from .worker import stitch


def main():
    repo=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=repo,check=True)
    with tempfile.TemporaryDirectory(prefix='lpam-bundled-') as temporary:
        info=stitch(repo/'Imgs/1_l.jpg',repo/'Imgs/1_r.jpg',Path(temporary)/'stitch_result.png')
        print(info,flush=True)
        if not info['lpam_triggered']: raise RuntimeError('Bundled smoke did not activate local patch alignment')


if __name__=='__main__': main()
