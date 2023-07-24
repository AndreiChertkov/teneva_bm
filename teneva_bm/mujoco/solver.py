class Solver:
    def __init__(self, env_name, opt_name, seed=42, steps=1000, policy_size=3):
        self.env_name = env_name
        self.opt_name = opt_name

        self.seed = seed
        self.steps = steps

        self.env = gym.make(self.env_name, render_mode='rgb_array')

        self.env_thr = self.env.spec.reward_threshold
        self.num_inputs = self.env.observation_space.shape[0]
        self.num_outputs = self.env.action_space.shape[0]

        self.policy = PolicyToeplitz(self.num_inputs, self.num_outputs)

        self.d = self.policy.d # Dimension of the Policy (for optimization)
        self.n = policy_size   # Mode size of the Policy (for optimization)

        self.theta = None      # Optimal found policy parameters
        self.reward = None     # Optimal found reward
        self.evals = 0         # Total number of calls from optimizer
        self.t = 0.            # Total optimization time

        # History of optimizer improvementes (evals and related opt values):
        self.evals_list = []
        self.reward_list = []

    def animate(self):
        reward, frames = self.run(self.theta, with_frames=True)
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        fpath = f'{self.env_name}_{self.opt_name}' + '.mp4'
        out = cv2.VideoWriter(fpath, fourcc, 20.0, (frames[0].shape[:2]))

        for frame in frames:
            out.write(frame)

        out.release()

    def info(self):
        text = '\n'
        text += '-' * 75 + '\n'
        text += f'Environment                    : {self.env_name}\n'
        text += f'Environment observation space  : {self.num_inputs}\n'
        text += f'Environment action space       : {self.num_outputs}\n'
        text += f'Dimension of the optimization  : {self.d}\n'
        text += f'Mode size for the optimization : {self.n}\n'
        text += f'Solver for the optimization    : {self.opt_name}\n'
        text += f'Selected random seed           : {self.seed}\n'
        text += '=' * 75 + '\n'
        print(text)

    def run(self, theta, with_frames=False):
        self.policy.set_theta(theta)

        state, info = self.env.reset(seed=int(self.seed))

        rewards = []
        frames = []
        done = False
        for step in range(self.steps):
            action = self.policy(state)
            state, reward, done, trunc, info = self.env.step(action)
            rewards.append(reward)

            if with_frames:
                frames.append(self.env.render())

            if done:
                break

        return (np.sum(rewards), frames) if with_frames else np.sum(rewards)

    def run_many(self, thetas):
        if len(thetas.shape) == 1:
            thetas = thetas[None]

        # return np.array([self.run(theta) for theta in thetas])

        rewards = []
        rewards = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(self.run)(theta) for theta in thetas)
        return np.array(rewards)

    def solve(self, evals=1.E+4, **opts):
        _time = tpc()

        self.evals = int(evals)

        if self.opt_name == 'htopt':
            self.theta, self.reward = self._solve_htopt(**opts)
        elif self.opt_name == 'protes':
            self.theta, self.reward = self._solve_protes(**opts)
        elif self.opt_name == 'random':
            self.theta, self.reward = self._solve_random(**opts)
        elif self.opt_name == 'ttopt':
            self.theta, self.reward = self._solve_ttopt(**opts)
        else:
            raise NotImplementedError(f'Not supported opt name {self.opt_name}')

        self.t = tpc() - _time

    def _solve_htopt(self):
        np.random.seed(self.seed)

        raise NotImplementedError

    def _solve_protes(self, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5):
        np.random.seed(self.seed)

        info = {}
        i, y = protes(self.run_many, self.d, self.n, self.evals, k, k_top, k_gd,
                lr, r, seed=self.seed, info=info, is_max=True, log=True)

        self.evals_list = info['m_opt_list']
        self.reward_list = info['y_opt_list']

        return i, y

    def _solve_random(self):
        np.random.seed(self.seed)

        t = tpc()
        thetas = np.random.choice(np.arange(self.n), (self.evals, self.d))
        rewards = self.run_many(thetas)
        k = np.argmax(rewards)
        i = thetas[k]
        y = rewards[k]
        for eval in range(self.evals):
            reward = rewards[eval]
            if len(self.reward_list) == 0 or self.reward_list[-1] < reward:
                self.evals_list.append(eval+1)
                self.reward_list.append(reward)
        t = tpc() - t

        print(f'random > m {self.evals:-7.1e} | t {t:-11.4e} | y {y:-11.4e}')

        return i, y

    def _solve_ttopt(self, r=5):
        np.random.seed(self.seed)

        Y0 = ttopt_init([self.n]*self.d, r)
        tto = TTOpt(f=self.run_many, d=self.d, n=self.n, evals=self.evals,
            name=self.env_name, is_func=False, with_log=True)
        tto.maximize(r, Y0=Y0)

        self.evals_list = np.cumsum(tto.evals_min_list)
        self.reward_list = np.array(tto.y_min_list)

        return tto.i_min, self.reward_list[-1]
