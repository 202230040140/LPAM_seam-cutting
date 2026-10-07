"""Build separate Windows DLLs from original kernels; generate I/O-only adapters."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'lpam_py' / '_native'


def main():
    def fingerprint(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    source_paths=[]
    for folder in ('SIFTflow/mexDenseSIFT','SIFTflow/mexDiscreteFlow','maxflow-v3.03.src'):
        source_paths.extend(p for p in (ROOT/folder).iterdir() if p.suffix in ('.h','.cpp','.inc'))
    source_paths.extend((ROOT/'lpam_py/native_src').glob('*.cpp'))
    signature={str(p.relative_to(ROOT)):fingerprint(p) for p in source_paths}
    adapter_hash=fingerprint(Path(__file__))
    previous_path=BUILD/'build_metadata.json'
    if previous_path.is_file():
        previous=json.loads(previous_path.read_text(encoding='utf-8'))
        if (previous.get('sources')==signature and previous.get('build_adapter_sha256')==adapter_hash
            and set(previous.get('dlls',{}))=={'dense.dll','flow.dll','cut.dll'}
            and all((BUILD/name).is_file() and fingerprint(BUILD/name)==value for name,value in previous['dlls'].items())):
            print('Verified existing original-kernel DLLs; no rebuild required.'); return
    vswhere = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    vs = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
                                  'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    if not vs:
        raise RuntimeError('MSVC x64 C++ toolchain is missing')
    env_text = subprocess.check_output(
        'call "' + vs + '\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul && set', shell=True, text=True)
    env = dict(os.environ)
    for line in env_text.splitlines():
        if '=' in line and not line.startswith('='):
            key,value=line.split('=',1); env[key]=value
    compiler=next(Path(vs).glob('VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))
    BUILD.mkdir(parents=True,exist_ok=True)
    metadata = {'compiler': vs, 'adapter': 'Disable MATLAB/Qt/OpenCV file I/O only; original numerical kernels retained',
                'build_adapter_sha256':adapter_hash,'sources': {}}
    for name,folder,extras in [('dense','SIFTflow/mexDenseSIFT',[]),
                               ('flow','SIFTflow/mexDiscreteFlow',['BPFlow.cpp','Stochastic.cpp']),
                               ('cut','maxflow-v3.03.src',[])]:
        stage=BUILD/name; stage.mkdir(exist_ok=True)
        for path in (ROOT/folder).iterdir():
            if path.suffix not in ('.h','.cpp','.inc'): continue
            source=path.read_text(encoding='utf-8-sig',errors='strict')
            metadata['sources'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            if name in ('dense','flow'):
                source=source.replace('#include "mex.h"','// MATLAB I/O omitted by standalone adapter')
                if path.name in ('Image.h','project.h'):
                    source=source.replace('#ifdef _MATLAB','#if 0 // standalone: no MATLAB array conversion')
                if path.name=='project.h':
                    source=source.replace('#define _LINUX_MAC','// MSVC provides __min/__max')
            (stage/path.name).write_text(source,encoding='utf-8')
        wrapper=ROOT/'lpam_py/native_src'/f'{name}.cpp'
        metadata['sources'][str(wrapper.relative_to(ROOT))]=hashlib.sha256(wrapper.read_bytes()).hexdigest()
        command=[str(compiler),'/nologo','/LD','/O2','/EHsc','/MD','/std:c++14','/D_MATLAB','/DNOMINMAX',
                 '/FItypeinfo','/I'+str(stage),str(wrapper)]+[str(stage/e) for e in extras]+['/link','/OUT:'+str(BUILD/f'{name}.dll')]
        subprocess.run(command,cwd=stage,env=env,check=True)
    metadata['dlls']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BUILD.glob('*.dll')}
    (BUILD/'build_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Built original Dense SIFT, BPFlow and maxflow kernels.')


if __name__=='__main__': main()
