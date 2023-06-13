import subprocess
import sys


def demo(bm_use=None):
    # Extract the file names of all benchmarks from the "__init__.py" file:
    with open('teneva_bm/__init__.py', encoding='utf-8') as f:
        lines = f.readlines()
    bms = []
    for l in lines:
        if 'import' in l and l[0] != '#':
            bm = l.split('from .')[1].split(' import')[0]
            if bm != 'bm':
                bms.append(bm)

    print(f'Full list of benchmark files (total: {len(bms)}):')
    print('>>>>', '; '.join(bms))
    print('\n\n')

    # Run the benchmark python file as the direct console call:
    for bm in bms:
        if bm_use and bm_use != bm:
            continue
            
        out = subprocess.getoutput(f'python teneva_bm/{bm}.py')
        print(out + '\n\n\n')
        if 'Traceback' in out:
            '\n\n<----- ERROR !!! Break.\n\n'
            return


if __name__ == '__main__':
    bm_use = sys.argv[1] if len(sys.argv) > 1 else None

    demo(bm_use)
