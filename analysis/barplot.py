import matplotlib.pyplot as plt


def single_loc_plot():
    """Plot results for single location model."""
    mfdgps = [0.600, 0.537, 0.371, 0.322]
    ml_benchs = [0.118, 0.117, -0.051, -0.118]
    env_benchs = [0.586, 0.439, -0.076]

    x_mfdgp = [0., 0.2, 0.4, 0.6]
    x_ml = [1., 1.2, 1.4, 1.6]
    x_env = [2., 2.2, 2.4]

    ticks = [0., 0.2, 0.4, 0.6, 1., 1.2, 1.4, 1.6, 2., 2.2, 2.4]
    tick_labels = ['Linear MFDGP (log)', 'Linear MFDGP (raw)',
                   'Nonlinear MFDGP (log)', 'Nonlinear MFDGP (raw)',
                   'GP (log)', 'GP (raw)', 'LinReg (log)',
                   'LinReg (raw)', 'APHRODITE',
                   'CRU', 'Bias-corrected WRF']

    plt.figure(figsize=(7, 7))

    plt.bar(x_mfdgp, mfdgps, color='#FFD700', width=0.15)
    for i, v in enumerate(mfdgps):
        plt.text(x_mfdgp[i] - .075, v + 0.01, '%.2f' % v)

    plt.bar(x_ml, ml_benchs, color='#D7D7D7', width=0.15)
    for i, v in enumerate(ml_benchs):
        if v > 0.0:
            plt.text(x_ml[i] - .075, v + 0.01, '%.2f' % v)
        if v < 0.0:
            plt.text(x_ml[i] - .075, v - 0.03, '%.2f' % v)

    plt.bar(x_env, env_benchs, color='#9BC3E1', width=0.15)
    for i, v in enumerate(env_benchs):
        if v > 0.0:
            plt.text(x_env[i] - .075, v + 0.01, '%.2f' % v)
        if v < 0.0:
            plt.text(x_env[i] - .075, v - 0.03, '%.2f' % v)

    plt.plot([-0.2, 2.6], [0, 0], linestyle='--', color='gray')
    plt.ylabel('$R^2$', fontsize=12)
    plt.xticks(ticks, labels=tick_labels, rotation=65)
    plt.show()


def multi_loc_plot():
    """Plot results for multiple location model."""
    mfdgps = [0.291, 0.206, -0.174, -0.421]
    ml_benchs = [-0.066, -0.084, -0.004, -0.040]
    env_benchs = [0.586, 0.439, -0.076]

    x_mfdgp = [0., 0.2, 0.4, 0.6]
    x_ml = [1., 1.2, 1.4, 1.6]
    x_env = [2., 2.2, 2.4]

    ticks = [0., 0.2, 0.4, 0.6, 1., 1.2, 1.4, 1.6, 2., 2.2, 2.4]
    tick_labels = ['Linear MFDGP (log)', 'Linear MFDGP (raw)',
                   'Nonlinear MFDGP (log)', 'Nonlinear MFDGP (raw)',
                   'GP (log)', 'GP (raw)', 'LinReg (log)',
                   'LinReg (raw)', 'APHRODITE',
                   'CRU', 'Bias-corrected WRF']

    plt.figure(figsize=(7, 7))

    plt.bar(x_mfdgp, mfdgps, color='#FFD700', width=0.15)
    for i, v in enumerate(mfdgps):
        if v > 0.0:
            plt.text(x_mfdgp[i] - .075, v + 0.01, '%.2f' % v)
        if v < 0.0:
            plt.text(x_mfdgp[i] - .075, v - 0.03, '%.2f' % v)

    plt.bar(x_ml, ml_benchs, color='#D7D7D7', width=0.15)
    for i, v in enumerate(ml_benchs):
        if v > 0.0:
            plt.text(x_ml[i] - .075, v + 0.01, '%.2f' % v)
        if v < 0.0:
            plt.text(x_ml[i] - .075, v - 0.03, '%.2f' % v)

    plt.bar(x_env, env_benchs, color='#9BC3E1', width=0.15)
    for i, v in enumerate(env_benchs):
        if v > 0.0:
            plt.text(x_env[i] - .075, v + 0.01, '%.2f' % v)
        if v < 0.0:
            plt.text(x_env[i] - .075, v - 0.03, '%.2f' % v)

    plt.plot([-0.2, 2.6], [0, 0], linestyle='--', color='gray')
    plt.ylabel('$R^2$', fontsize=12)
    plt.xticks(ticks, labels=tick_labels, rotation=65)
    plt.show()
