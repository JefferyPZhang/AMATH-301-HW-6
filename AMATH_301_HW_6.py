import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

# Problem 1
def adam_moulton(f, t_span, y_0):
    A1 = 0
    sol = np.zeros(len(t_span))
    sol[0] = y_0
    step = t_span[1] - t_span[0]
    for i in range (1, len(sol)):
        g = lambda z: (sol[i - 1] + (step / 2) * (f(t_span[i], z) + (f(t_span[i - 1], sol[i - 1])))) - z
        if (i == 1): A1 = abs(g(3))
        sol[i] = scipy.optimize.fsolve(g, sol[i - 1])[0]
    return A1, sol

dydt = lambda t, y: (5 * 10 ** 5) * (np.sin(t) - y)
A1, A2 = adam_moulton(dydt, np.linspace(0, 2 * np.pi, 100), 0)

# Problem 2

k = 0
q_val = np.logspace(0, -5, 10)
param = [77.27, 0.161, q_val[k]]
y_0 = [1, 2, 3]

odefun = lambda t, y: np.array((param[0] * (y[1] - y[0] * y[1] + y[0] - param[2] * y[0] ** 2),
                       (-y[1] - y[0] * y[1] + y[2]) / param[0],
                       param[1] * (y[0] - y[2])))

A3 = odefun(1, [2, 3, 4])

sol = np.zeros((3, 10))
for i in range (10):
    param = [77.27, 0.161, q_val[i]]
    odefun = lambda t, y: np.array((param[0] * (y[1] - y[0] * y[1] + y[0] - param[2] * y[0] ** 2),
                       1/ param[0]*(-y[1] - y[0] * y[1] + y[2]) ,
                       param[1] * (y[0] - y[2])))
    iter_sol = scipy.integrate.solve_ivp(odefun, [0, 30], y_0)
    sol[:, i] = iter_sol.y[:, -1]
    k += 1
A4 = sol
print(A4)
k = 0
for i in range (10):
    param = [77.27, 0.161, q_val[k]]
    iter_sol = scipy.integrate.solve_ivp(odefun, [0, 30], y_0, method = 'BDF')
    sol[:, i] = iter_sol.y[:, -1]
    k += 1  
A5 = sol

# Question 3

mu = 200
v_0 = np.array([2, 0])
dxdt = lambda t, x, y: y
dydt = lambda t, x, y: mu * (1 - x ** 2) * y - x
dvdt = lambda t, v: np.array([v[1], mu * (1 - v[0] ** 2) * v[1] - v[0]])

A6 = dxdt(0, 2, 3)
A7 = dydt(0, 2, 3)
RK45_sol = scipy.integrate.solve_ivp(dvdt, [0, 400], v_0)
BDF_sol = scipy.integrate.solve_ivp(dvdt, [0, 400], v_0, method = 'BDF')
A8 = RK45_sol.y[0]
A9 = BDF_sol.y[0]
A10 = len(A8) / len(A9)
print(A10)
dydt = lambda t, x, y: mu * y - x
dvdt = lambda t, v: np.array([v[1], mu * v[1] - v[0]])
A11 = A6
A12 = dydt(0, 2, 3)

plt.scatter(RK45_sol.t, RK45_sol.y[0])
plt.scatter(BDF_sol.t, BDF_sol.y[0])
plt.xlabel("t-val")
plt.ylabel("x-val")
plt.title("Solution Graph of RK45 & BDF solve_ivp Methods, Compared")
plt.legend(("RK45", "BDF"))
plt.show()

plt.figure()
plt.plot(BDF_sol.y[0], BDF_sol.y[1])
plt.xlabel("x-val")
plt.ylabel("y-val")
plt.title("Solution Points of BDF solve_ivp Method")
plt.show()

A = np.array([[0, 1], [-1, mu]])
A13 = A
C = np.eye(2) - 0.01 * A
A16 = C
C_inv = np.linalg.inv(C)
t_span = np.arange(0, 400.01, 0.01)
step = 0.01
    
def forward_euler(A, t_span, x_0):
    sol = np.zeros((2, len(t_span)))
    sol[:, 0] = x_0
    for i in range (1, len(t_span)):
        sol[:, i] = sol[:, i - 1] + step * A @ sol[:, i - 1]
    return sol

def backward_euler(C, t_span, x_0):
    sol = np.zeros((2, len(t_span)))
    sol[:, 0] = x_0
    for i in range (1, len(t_span)):
        sol[:, i] = C_inv @ sol[:, i - 1]
    return sol

A14 = forward_euler(A, t_span, v_0)[0]
A15 = A14[:10]
A17 = backward_euler(C, t_span, v_0)[0]