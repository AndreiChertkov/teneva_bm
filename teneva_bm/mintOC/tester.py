import matplotlib.pyplot as plt
from protes import protes
import protes_bms
import gekko_bms

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('problem', type=str, help='name of a problem')
parser.add_argument('d', type=int, help='dimension')
parser.add_argument('m', type=int, help='budget')
args = parser.parse_args()


def plot_gekko_vs_protes(bm, solution_gekko, solution_protes):
    control_var_gekko, obj_gekko = solution_gekko
    control_var_protes, obj_protes = solution_protes
    
    state_var_gekko = bm._ode(control_var_gekko)['x']
    state_var_protes = bm._ode(control_var_protes)['x']
    n_vars = state_var_protes.shape[0]
    time = bm.t

    print(f' GEKKO: {obj_gekko:.4f}')
    print(f'PROTES: {obj_protes:.4f}')

    _, axes = plt.subplots(n_vars+1, 1, sharex=True, figsize=(12, (n_vars+1)*3))
    plt.xlabel('Time')
    
    plt.subplot(n_vars+1, 1, 1)
    axes[0].set_title('Control Variable')
    axes[0].plot(time, control_var_gekko, label='GEKKO')
    axes[0].plot(time, control_var_protes, label='PROTES')
    axes[0].legend()

    for i in range(n_vars):
        plt.subplot(n_vars+1, 1, i+2)
        axes[i+1].set_title(f'State Variable {i+1}')
        axes[i+1].plot(time, state_var_gekko[i], label='GEKKO')
        axes[i+1].plot(time, state_var_protes[i], label='PROTES')
        axes[i+1].legend()

    plt.show()

def protes_solver(bm, m):
    _, obj = protes(bm.get, bm.d, bm.n0, m=m, log=False)
    control_var = bm.x_min if bm.is_func else bm.i_min
    return control_var, obj

def main(problem, d, m):
    """
    GEKKO vs PROTES
    """
    protes_class_name = 'Bm' + ''.join([w.capitalize() for w in problem.split('_')])
    try:
        bm = getattr(protes_bms, protes_class_name)(d=d).prep()
        gekko_solver = getattr(gekko_bms, problem)
        solution_gekko = gekko_solver(d, m)
        solution_protes = protes_solver(bm, m)
        plot_gekko_vs_protes(bm, solution_gekko, solution_protes)
    except:
        print('Wrong problem name.')

if __name__ == '__main__':
    main(args.problem, args.d, args.m)