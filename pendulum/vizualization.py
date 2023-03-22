def animate_pendulum(x,y,vx=None,vy=None,save=None):
    """
    Animates the pendulum.
    If save is None it will show a movie in python
    otherwise it will save a gif to the location specified in save
    """

    import matplotlib;
    matplotlib.use("TkAgg")
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    n,nsteps = x.shape

    line, = ax.plot(x[:,0], y[:,0])
    line2, = ax.plot(x[:,0], y[:,0], 'ro')
    lines = [line,line2]

    v_origo =np.asarray([x[1:,0], y[1:,0]])
    arrows = plt.quiver(*v_origo, vx[1:, 0], vy[1:, 0], color='r', scale=100,width=0.003)
    lines.append(arrows)

    def init():
        ax.set_xlim(-n,n)
        ax.set_ylim(-n,n)
        return lines

    def update(i):
        for j in range(2):
            lines[j].set_xdata(x[:,i])
            lines[j].set_ydata(y[:,i])
        origo = np.asarray([x[1:, i], y[1:, i]]).transpose()
        lines[-1].set_offsets(origo)
        lines[-1].set_UVC(vx[1:, i], vy[1:, i])
        return lines

    ani = animation.FuncAnimation(fig, update, frames=np.arange(1, nsteps), init_func=init, interval=25, blit=True)
    if save is None:
        plt.show()
    else:
        writergif = animation.PillowWriter(fps=30)
        ani.save(save, writer=writergif)

