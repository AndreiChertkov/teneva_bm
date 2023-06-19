import subprocess
import sys


def demo(bm_use=None):
    # Extract the file names of all benchmarks from the "__init__.py" file:
    with open('teneva_bm/__init__.py', encoding='utf-8') as f:
        lines = f.readlines()
    bms = []
    modules = []
    for l in lines:
        if 'import' in l and l[0] != '#':
            bm_full = l.split('from .')[1].split(' import')[0]
            if not '.' in bm_full:
                continue

            bms.append(bm_full.split('.')[1])
            modules.append(bm_full.split('.')[0])

    print(f'Full list of benchmark files (total: {len(bms)}):')
    print('>>>>', '; '.join(bms))
    print('\n\n')

    # Run the benchmark python file as the direct console call:
    for bm, module in zip(bms, modules):
        if bm_use and bm_use != bm:
            continue

        out = subprocess.getoutput(f'python teneva_bm/{module}/{bm}.py')
        print(out + '\n\n\n')
        if 'Traceback' in out:
            '\n\n<----- ERROR !!! Break.\n\n'
            return


if __name__ == '__main__':
    bm_use = sys.argv[1] if len(sys.argv) > 1 else None

    demo(bm_use)
