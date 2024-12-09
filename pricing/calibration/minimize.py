from scipy import optimize

def minimize(fun, x0=None, method=None, **kwargs):
    def callback(x, f=None, context=None, accept=None, convergence=None):
        val = fun(x)
        print(f"New iteration: \n x = {x}, \n val={val}. \n")
        if convergence is not  None:
            print(f"Convergence = {convergence}")
        if val <= 1e-6:
            raise StopIteration

    if "callback" not in kwargs:
        kwargs["callback"] = callback

    if method=="Differential-evolution":
        res = optimize.differential_evolution(func=fun, x0=x0, **kwargs)
    else:
        res = optimize.minimize(fun=fun, x0=x0, method=method, **kwargs)

    return res