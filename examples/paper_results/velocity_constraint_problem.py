import torch


def delta_r(r):
    """
    Computes a vector from each pendulum to the next, including origo.
    """
    dr_0 = r[:, 0]
    dr_i = r[:, 1:] - r[:, :-1]
    dr = torch.cat((dr_0[:, None], dr_i), dim=1)
    return dr


def extract_positions(x):
    """
    Extracts positions, r, from x
    """
    position_idx = [0, 1]
    r = x[:, :, position_idx]
    return r


def extract_velocity(x):
    """
    Extracts velocities, v, from x
    """
    velocity_idx = [2, 3]
    v = x[:, :, velocity_idx]
    return v


def constraint_1(x):
    """
    Computes the constraints.

    For a n multi-body pendulum the constraint can be given as:
    c_i = (r_i - r_{i-1})**2 - l_i**2,   i=1,n

    Note that this should not be used when computing the constraint violation since that should be done with the unsquared norm.
    """
    r = extract_positions(x)
    dr = delta_r(r)
    dr2 = (dr*dr).sum(dim=-1)
    c = dr2 - 1
    return c[:,:,None]

def constraint_2(r,v):
    """
    Bilinear velocity constraints
    """
    dr = delta_r(r)
    dv = delta_r(v)
    c = 2*(dr*dv).sum(dim=-1)

    return c[:,:,None]


def f(x):
    c1 = constraint_1(x)

    v = extract_velocity(x)
    r = extract_positions(x)
    c2 = constraint_2(r,v)
    c = torch.cat((c1, c2), dim=-1)
    return c
def steepest_descent(f,x):
    """
    """
    imax = 100
    tol = 1e-5
    i = 0
    while i < imax:
        _, JTc = torch.autograd.functional.vjp(f, x, f(x), strict=True, create_graph=True)
        c = f(x)
        alpha = 1
        while True:
            x_try = x - alpha * JTc
            c_try = f(x_try)
            print(f"alpha={alpha:2.2e}  {c.abs().mean():2.2e} -> {c_try.abs().mean():2.2e}")
            if c.abs().mean() > c_try.abs().mean() or alpha==0:
                x = x_try
                break
            else:
                alpha *= 0.5
        if c_try.abs().mean() < tol:
            break
        i += 1
    c = f(x)
    print(f"{i},  {c.abs().mean():2.2e}")
    return x

import matplotlib.pyplot as plt





def plot_pendulum(x):
    r = extract_positions(x)
    v = extract_velocity(x)
    r = r.squeeze()
    v = v.squeeze().numpy()
    fig, ax = plt.subplots(figsize=(15,15))
    origo = torch.tensor([[0.0,0.0]])
    rext = torch.cat((origo,r),dim=0)
    rext = rext.numpy()
    v_origo = rext[1:].T
    plt.quiver(*v_origo, v[:, 0], v[:, 1], color='r', scale=200,width=0.003, alpha=0.2)
    l_p, = ax.plot(rext[:, 0], rext[:, 1], '--', color='lime', alpha=0.7)
    lm_p, = ax.plot(rext[:, 0], rext[:, 1], 'go', alpha=0.7)
    plt.axis('square')
    plt.pause(1)


if __name__ == '__main__':
    # A 5 body pendulum with parametrized as: (x,y,vx,vy)
    x = torch.tensor([[[-0.5645338846, -0.8254172212, 2.0822525369, -1.4241232364],
             [-1.3649010534, -1.4249110136, -2.6331244370, 4.8700485503],
             [-0.7149567356, -2.1848565012, -5.4748682974, 2.4394292257],
             [-1.6721357614, -2.4745400104, -2.8997539725, -6.0740265586],
             [-2.6529855450, -2.6689519757, -2.5279726930, -7.9494982741]]])

    plot_pendulum(x)
    x2 = steepest_descent(f,x)
    print("done")