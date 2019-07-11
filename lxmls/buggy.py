def next_x(x):
    xnext = x + np.random.normal(scale=.0625)
    if xnext < 0:
        return 0.
    return xnext


def walk():
    iters = 0
    x = 0
    while x <= 1.:
        x = next_x(x)
        iters += 1
    return iters


walks = np.array([walk() for i in range(1000)])

print(np.mean(walks))