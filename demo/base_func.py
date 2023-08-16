"""Simple example of benchmark usage for "BmFuncAckley".

To run the code use the following command:
$ clear && python demo/base_func.py

"""
import numpy as np
from teneva_bm import BmFuncAckley


def demo():
    bm = BmFuncAckley().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+4)
    print(bm.info_history())

    text = 'Value at a random multi-index            :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Value at 3 random multi-indices          :  '
    i1 = [np.random.choice(k) for k in bm.n]
    i2 = [np.random.choice(k) for k in bm.n]
    i3 = [np.random.choice(k) for k in bm.n]
    I = [i1, i2, i3]
    y = bm[I]
    text += '; '.join([f'{y_cur:-10.3e}' for y_cur in y])
    print(text)

    text = 'Value at the minimum (real vs calc)      :  '
    y_real = bm.y_min_real
    y_calc = bm(bm.x_min_real)
    text += f'{y_real:-10.3e}       /      {y_calc:-10.3e}'
    print(text)

    text = 'Value at the ref multi-index             :  '
    bm = BmFuncAckley().prep()
    i_ref, y_ref = bm.ref
    y = bm[i_ref]
    text += f'{y:-14.7e} [{y_ref:-14.7e}]'
    print(text)

    print('\n\n\n')


if __name__ == '__main__':
    np.random.seed(42)
    demo()
