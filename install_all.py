"""Python script to install additional dependencies for teneva_bm.

This script may be used for installation of additional dependencies (including
gym, mujoco and several other libs) on linux, osx and colab platforms. You
should set the name of existing conda environment as an argument (except the
colab platform), e.g., "--env teneva_bm", and installation will be done in the
related environment.

* The correctness of the script was verified, including on noisy and zhores
clusters of our group. If you run this script on zhores, then you should set
the flag "--zhores" (note that rendering does not work in this case now).

For convenience, the script can be downloaded with the following command:
$ wget https://raw.githubusercontent.com/AndreiChertkov/teneva_bm/main/install_all.py

An existing environment can be specified as an argument to the script, or a new
one can be pre-created:
$ conda create --name teneva_bm python=3.8 -y

* If the environment already exists, you can (optionally) delete it before:
$ conda activate && conda remove --name teneva_bm --all -y

* In the case of zhores, you should run before environment creation the command:
$ module load python/anaconda3

>>>>> To run the script, use the following command (in the case of a colab
platform, you should not set the environment; also note that you does not need
to activate the environment manually before run the script):
$ clear && python install_all.py --env teneva_bm

* If there are any problems with the script, you can run it with "--log" flag
(full output from all commands will be presented) and check the outputs. If you
want to remove all messages (except errors) presented by the script, then you
can specify the flag "--silent".

>>>>> To check the success of the script, you can activate the environment and
run a test computation (the video will be generated after computation):
$ conda activate teneva_bm && clear && python install_all.py --test

* In the case of zhores cluster you should set the flag "--zhores", the video
will not be generated in this case.

"""
import argparse
import os
import subprocess
import sys


WITH_INFO = True
WITH_LOG = False
PLATFORMS = ['linux', 'osx', 'colab', 'zhores']
PY_PACKAGES = [
    'mujoco==2.3.6',
    'imageio==2.31.1',
    'gym==0.26.2',
    'opencv-python==4.7.0.72',
    'pygame==2.1.0',
    'swig==4.1.1',
    'box2d-py==2.3.5',
    'networkx==3.0',
    'qubogen==0.1.1',
    'gekko==1.0.6']
PY_PACKAGES_COLAB = [
    'PyOpenGL_accelerate',
    'free-mujoco-py==2.1.6',
    'gym==0.26.2',
    'mujoco==2.3.7',
    'imageio==2.31.1',
    'opencv-python==4.7.0.72',
    'pygame==2.1.0',
    'swig==4.1.1',
    'box2d-py==2.3.5',
    'networkx==3.0',
    'qubogen==0.1.1',
    'gekko==1.0.6']


def install_all(env=None, with_info=True, with_log=False, is_zhores=False):
    global WITH_INFO, WITH_LOG
    WITH_INFO, WITH_LOG = with_info, with_log

    _log('Check that platform (OS) is supported', 'PRC')
    if is_zhores:
        platform = 'zhores'
    else:
        platform = _check_platform()
    if not platform in PLATFORMS:
        return _log(f'The platform "{platform}" is not supported', 'ERR')
    _log(f'Supported platform "{platform}" is detected')

    if not env and platform != 'colab':
        return _log(f'Arg "env" is required for platform "{platform}"', 'ERR')
    if env and platform == 'colab':
        return _log(f'Arg "env" can not be set for colab', 'ERR')

    if env:
        _log('Check the environment', 'PRC')
        if not _check_env(env):
            return _log(f'Can not use the environment "{env}"', 'ERR')
        _log(f'Environment "{env}" is valid')

    if platform == 'colab':
        install_all_colab()
    if platform == 'linux':
        install_all_linux(env)
    if platform == 'osx':
        install_all_osx(env)
    if platform == 'zhores':
        install_all_zhores(env)

    msg = 'Work is finished'
    if platform != 'colab':
        msg += '. Please activate your environment as '
        msg += f'"conda activate {env}" and use the teneva_bm package...'

    msg += '\n... you can check the result as "'
    if env:
        msg += f'conda activate {env} && '
    msg += 'python install_all.py --test'
    if platform == 'zhores':
        msg += ' --zhores'
    msg += '"'
    _log(msg)


def install_all_colab():
    GL_LIBS_COLAB = ['libgl1-mesa-dev', 'libgl1-mesa-glx', 'libglew-dev',
        'libosmesa6-dev', 'xpra', 'patchelf', 'libglfw3-dev']

    for lib in GL_LIBS_COLAB:
        _log(f'Install library "{lib}" for GL', 'PRC')
        res, out = _run(f'apt-get install -y {lib}')
        if False: # TODO: check
            return _log(f'Can not install library "{lib}" for GL', 'ERR')
        _log(f'Installed library "{lib}" for GL')

    _log('Remove unused python package "dopamine"', 'PRC')
    res, out = _run('! pip uninstall dopamine-rl -y')
    _log('Unused python package "dopamine" is removed')

    for pack in PY_PACKAGES_COLAB:
        _log(f'Install python package "{pack}"', 'PRC')
        res, out = _run(f'pip install {pack}')
        _log(f'Installed python package "{pack}"')

    _log(f'Build python package "mujoco-py"', 'PRC')
    res, out = _run('python -c "import mujoco_py"')
    _log(f'Python package "mujoco-py" is ready')

    _log('Edit GLFW lib to remove annoying warnings', 'PRC')
    if _run_glfw_edit(fold='/usr/local/lib/python3.10/dist-packages'):
        _log(f'GLFW lib is edited')
    else:
        _log(f'GLFW lib was not (!) edited', 'WRN')


def install_all_linux(env):
    GL_LIBS_LINUX = [
        ['glew', 'conda-forge'],
        ['mesalib', 'conda-forge'],
        ['mesa-libgl-cos6-x86_64', 'anaconda'],
        ['glfw3', 'menpo']]

    for [lib, repo] in GL_LIBS_LINUX:
        _log(f'Install library "{lib}" for GL', 'PRC')
        res, out = _run(f'conda install -c {repo} {lib} -y', env=env)
        if False: # TODO: check
            return _log(f'Can not install library "{lib}" for GL', 'ERR')
        _log(f'Installed library "{lib}" for GL')

    res, out = _run(
        'conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl',
        env=env)

    _log('Download mujoco', 'PRC')
    res, out = _run('rm -r ~/.mujoco')
    res, out = _run(
        'wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz',
        'tar -xf mujoco210-linux-x86_64.tar.gz',
        'rm mujoco210-linux-x86_64.tar.gz',
        'mkdir ~/.mujoco',
        'mv mujoco210 ~/.mujoco/')
    res, out = _run('rm -r mujoco210')
    _log('Mujoco is downloaded')

    for pack in PY_PACKAGES:
        _log(f'Install python package "{pack}"', 'PRC')
        res, out = _run(f'pip install {pack}', env=env)
        _log(f'Installed python package "{pack}"')

    if False:
        _log('Edit OpenGL lib to remove annoying warnings', 'PRC')
        if _run_opengl_edit(env):
            _log(f'OpenGL lib is edited')
        else:
            _log(f'OpenGL lib was not (!) edited', 'WRN')


def install_all_osx(env):
    _log('Download mujoco', 'PRC')
    res, out = _run(
        'wget https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz',
        'tar -xf mujoco210-macos-x86_64.tar.gz',
        'rm mujoco210-macos-x86_64.tar.gz',
        'mkdir ~/.mujoco',
        'mv mujoco210 ~/.mujoco/')
    res, out = _run('rm -r mujoco210')
    _log('Mujoco is downloaded')

    for pack in PY_PACKAGES:
        _log(f'Install python package "{pack}"', 'PRC')
        res, out = _run(f'pip install {pack}', env=env)
        _log(f'Installed python package "{pack}"')

    if False:
        _log('Edit OpenGL lib to remove annoying warnings', 'PRC')
        if _run_opengl_edit(env):
            _log(f'OpenGL lib is edited')
        else:
            _log(f'OpenGL lib was not (!) edited', 'WRN')

    _log('Edit GLFW lib to remove annoying warnings', 'PRC')
    if _run_glfw_edit(env):
        _log(f'GLFW lib is edited')
    else:
        _log(f'GLFW lib was not (!) edited', 'WRN')


def install_all_zhores(env):
    _log('Download mujoco', 'PRC')
    res, out = _run('rm -r ~/.mujoco')
    res, out = _run(
        'wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz',
        'tar -xf mujoco210-linux-x86_64.tar.gz',
        'rm mujoco210-linux-x86_64.tar.gz',
        'mkdir ~/.mujoco',
        'mv mujoco210 ~/.mujoco/')
    res, out = _run('rm -r mujoco210')
    _log('Mujoco is downloaded')

    for pack in PY_PACKAGES:
        _log(f'Install python package "{pack}"', 'PRC')
        res, out = _run(f'pip install {pack}', env=env)
        _log(f'Installed python package "{pack}"')


def test(with_video=True):
    import cv2
    import gym
    import numpy as np
    np.bool8 = bool

    env = gym.make('Swimmer-v4', render_mode='rgb_array')
    env.reset(seed=42)

    state = np.zeros(8)
    qpos = np.array([0., 0.] + list(state[:3]))
    qvel = state[3:]
    env.set_state(qpos, qvel)

    frames = []
    for step in range(10):
        action = env.action_space.sample()
        state, reward = env.step(action)[:2]
        print(f'Step: {step+1:-3d} | Reward: {reward:-8.1e}')

        if with_video:
            frames.append(env.render())

    if with_video:
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        fpath = 'video_install_all.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fpath, fourcc, 20.0, (frames[0].shape[:2]))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f'DONE      | Video is saved into "{fpath}"')
    else:
        print(f'DONE      |')


def _args_build():
    parser = argparse.ArgumentParser(
        prog='install_all',
        description='Python script to install dependencies for teneva_bm',
        epilog = '© Andrei Chertkov'
    )
    parser.add_argument('--env',
        type=str,
        help='Name of the existing conda environment to use'
    )
    parser.add_argument('--silent',
        action='store_false',
        help='Do not print info about the working process'
    )
    parser.add_argument('--log',
        action='store_true',
        help='Print full logs while working process'
    )
    parser.add_argument('--zhores',
        action='store_true',
        help='Set this flag if use zhores cluster'
    )
    parser.add_argument('--test',
        action='store_true',
        help='Set this flag to check the result of installation'
    )
    args = parser.parse_args()
    return (args.env, args.silent, args.log, args.zhores, args.test)


def _check_env(env):
    res, out = _run(
        'echo "Start install_all script (check environment)..."',
        env=env, check_false='EnvironmentLocationNotFound')

    if not res:
        msg = f'Environment "{env}" does not exist. '
        msg += f'Run "conda create --name {env} python=3.8 -y" before'
        _log(msg, 'WRN')
        return False

    return True


def _check_platform():
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False

    if is_colab:
        return 'colab'

    if sys.platform in ['linux', 'linux2']:
        return 'linux'

    if sys.platform == 'darwin':
        return 'osx'

    if sys.platform == 'win32':
        return 'windows'


def _log(msg, kind='RES'):
    if kind == 'ERR':
        print(f'!!! ERROR !!! {msg}')
    elif kind == 'WRN':
        print(f'!!! {msg}')
    elif kind == 'PRC' and WITH_INFO:
        print(f'... {msg}')
    elif kind == 'RES' and WITH_INFO:
        print(f'>>> {msg}\n')
    elif WITH_INFO:
        print(f'>>> [{kind}] {msg}')


def _run(*cmd, env=None, check_true=None, check_false=None, cmd_join=' && '):
    cmd = cmd_join.join(cmd)
    cmd = f'conda run -n {env} {cmd}' if env else cmd
    out = subprocess.getoutput(cmd)

    if WITH_LOG:
        print('\n\n' + out + '\n\n')

    if check_true:
        return (check_true in out), out

    if check_false:
        return not (check_false in out), out

    return True, out


def _run_opengl_edit(env=None, fold=None):
    if not fold:
        res, out = _run('which python', env=env, check_true='anaconda')
        if not res:
            res, out = _run('which python3', env=env, check_true='anaconda')
            if not res:
                _log('Can not find anaconda path', 'WRN')
                return False
        fold = out.split('/envs/')[0]
        fold += f'/envs/{env}/lib/python3.8/site-packages'

    fpath = fold + '/OpenGL/error.py'

    try:
        with open(fpath, 'r') as f:
            text = f.read()
    except Exception as e:
        _log('Can not find OpenGL lib folder', 'WRN')
        return False

    line = 'err != self._noErrorResult:'
    if not f'if {line}' in text:
        return True

    text = text.replace(line, 'False and ' + line)
    with open(fpath, 'w') as f:
        f.write(text)


def _run_glfw_edit(env=None, fold=None):
    if not fold:
        res, out = _run('which python', env=env, check_true='anaconda')
        if not res:
            res, out = _run('which python3', env=env, check_true='anaconda')
            if not res:
                _log('Can not find anaconda path', 'WRN')
                return False
        fold = out.split('/envs/')[0]
        fold += f'/envs/{env}/lib/python3.8/site-packages'

    fpath = fold + '/mujoco/glfw/__init__.py'

    try:
        with open(fpath, 'r') as f:
            text = f.read()
    except Exception as e:
        _log('Can not find mujoco lib folder', 'WRN')
        return False

    line = 'def free(self):'
    line_new = line + '\n    self._context = None\n    return'
    if not line in text:
        return True

    text = text.replace(line, line_new)
    with open(fpath, 'w') as f:
        f.write(text)

    return True


if __name__ == '__main__':
    env, with_info, with_log, is_zhores, is_test = _args_build()

    if is_test:
        test(with_video=not is_zhores)
    else:
        install_all(env, with_info, with_log, is_zhores)
