import os
import re
import subprocess


def teneva_bm_demo(bm_use=None, with_info=True):
    """Demo script for teneva_bm.

    Run demo for all existing benchmarks (if "bm_use" is None) or for a
    specific benchmark (if "bm_use" is provided; e.g., "bm_qubo_knap_amba"). If
    the flag "with_info" is set, then the stats for existing collections and
    benchmarks will be also printed. Note that the code for the demo run of
    benchmarks should be at the end of the corresponding benchmark file in the
    section "if __name__ == '__main__':".

    """
    bms = {}
    found = False

    bm_use = _parse_bm_name(bm_use)

    for cl in _find_cl_all():
        bms[cl] = _find_bm_all(cl)
        if bm_use in bms[cl]:
            found = True

    if with_info:
        _info(bms)

    if bm_use and not found:
        msg = f'Benchmark "{bm_use}" does not exist. '
        msg += 'Can not run demo'
        raise ValueError(msg)

    _run_all(bms, bm_use)


def _find_bm_all(cl):
    """Collect all existing benchmark names for the provided collection name."""
    bms = []
    here = os.path.abspath(os.path.dirname(__file__))
    with open(f'{here}/{cl}/__init__.py', encoding='utf-8') as f:
        lines = f.readlines()
    for l in lines:
        if 'from .' in l and ' import ' in l and l[0] != '#':
            bms.append(l.split('from .')[1].split(' import')[0])
    return bms


def _find_cl_all():
    """Collect all existing collection names."""
    cls = []
    here = os.path.abspath(os.path.dirname(__file__))
    with open(f'{here}/__init__.py', encoding='utf-8') as f:
        lines = f.readlines()
    for l in lines:
        if 'from .' in l and ' import *' in l and l[0] != '#':
            cls.append(l.split('from .')[1].split(' import *')[0])
    return cls


def _info(bms):
    """Present the stats for existing collections and benchmarks."""
    text = '\n\n\n' + '-' * 70 + '\n'
    text += '-' * 19 + ' Benchmarks library (teneva_bm) ' + '-' * 19 + '\n'
    text += '-' * 70 + '\n\n'

    for cl, bm_list in bms.items():
        count = len(bm_list)
        text += f'--> {cl}' + ' ' * max(0, 10-len(cl))
        text += f': {count:-4d} benchmarks'
        for i in range(count):
            if i == 0 or i % 3 == 0:
                text += '\n' + ' ' * 14 + '> '
            text += str(bm_list[i])
            if i < count - 1:
                text += ', '
        text += '\n'

    text += '\n' + '=' * 70 + '\n\n\n\n'

    print(text)


def _parse_bm_name(name):
    """Converts a name of the benchmark to underscore notation."""
    if name:
        if not name.lower().startswith('bm'):
            name = 'bm_' + name[0].lower() + name[1:]
        return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()


def _run_all(bms, bm_use=None):
    """Run demo for all benchmarks of only for "bm_use" if provided."""
    here = os.path.abspath(os.path.dirname(__file__))

    for cl, bm_list in bms.items():
        for bm in bm_list:
            if bm_use and bm_use != bm:
                continue

            path = f'{here}/{cl}/{bm}.py'
            out = subprocess.getoutput(f'python {path}')
            print(out + '\n\n')

            if 'Traceback' in out:
                '\n\n<----- ERROR !!! Break.\n\n'
                return
