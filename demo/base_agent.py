"""Simple example of benchmark usage for "BmAgentSwimmer".

To run the code use the following command:
$ clear && python demo/base_agent.py
The generated video and image for the one (ref) strategy will be saved as a
files "result/demo/base_agent.[mp4/png]".

"""
import numpy as np
from teneva_bm import BmAgentSwimmer


def demo():
    bm = BmAgentSwimmer().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+1)
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

    text = 'Value at the ref multi-index             :  '
    bm = BmAgentSwimmer().prep()
    i_ref, y_ref = bm.ref
    y = bm[i_ref]
    text += f'{y:-14.7e} [{y_ref:-14.7e}]'
    print(text)

    bm.render(f'result/demo/base_agent', best=False)
    bm.show(f'result/demo/base_agent', best=False)

    print('\n\n\n')


if __name__ == '__main__':
    np.random.seed(42)
    demo()
