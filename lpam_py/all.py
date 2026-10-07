"""Build, smoke, then run all three manifests; failures remain explicit."""
import subprocess
import sys


def main():
    for module in ('lpam_py.build_native','lpam_py.smoke'):
        subprocess.run([sys.executable,'-m',module],check=True)
    tests=[('stitchbench','AANAP-intersection'),('stitchbench','OBJ-GSP-tree2'),
           ('stitchbench','LPC-06'),('stitchbench','SVA-01_chess'),('hd3d','Indoor_001'),('lpsd',None)]
    smoke_success=set()
    for benchmark,scene in tests:
        command=[sys.executable,'-m','lpam_py.run','--benchmark',benchmark,'--limit','1','--skip-existing','--stop-on-error']
        if scene: command+=['--scene',scene]
        result=subprocess.run(command)
        if result.returncode==0: smoke_success.add(benchmark)
        else: print(f'Smoke failure recorded: {benchmark}/{scene}; no substitute output.',flush=True)
    if smoke_success!={'stitchbench','hd3d','lpsd'}:
        raise SystemExit('No successful smoke in: '+', '.join({'stitchbench','hd3d','lpsd'}-smoke_success))
    failures=[]
    for benchmark in ('stitchbench','hd3d','lpsd'):
        result=subprocess.run([sys.executable,'-m','lpam_py.run','--benchmark',benchmark,'--skip-existing']+sys.argv[1:])
        if result.returncode: failures.append(benchmark)
    subprocess.run([sys.executable,'-m','lpam_py.audit'],check=True)
    if failures: raise SystemExit('Recorded failures: '+', '.join(failures))


if __name__=='__main__': main()
