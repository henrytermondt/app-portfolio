"""
Quantum Wave Function
24/7/23

Resources:
http://www.astro.utoronto.ca/~mahajan/notebooks/quantum_tunnelling.html

Barrier:
With n = 1000, dt = 0.1 and 1000 frames, it takes 25.92 minutes when using the PDF
With n = 1000, dt = 0.1 and 1000 frames, it takes 25.4 minutes without the PDF

Simple:
With n = 1000, dt = 0.1 and 1000 frames, it takes 25.67 minutes without the PDF
"""

import numpy as np
import scipy.sparse
import scipy.linalg
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

startTime = time.time()

usePDF = True

start = -80
end = 80
n = 1000
width = end - start
x, dx = np.linspace(start, end, n, endpoint=False, retstep=True)
scale = n / width

def getRoot(t, e, v0):
    x = np.sqrt(((1 - t) * 4 * e * (v0 - e) / (t * v0 * v0)))
    return np.log(x + np.sqrt(x * x + 1)) / np.sqrt(2 * (v0 - e))
def createBarrier(x, w, h):
    v = []
    for i in range(n):
        curx = i / scale + start
        v.append(h if curx > x and curx < x + w else 0)
    return v

t = 0.2
r = 3 / 4
k1 = getRoot(t, 0.5 * r, 0.5)
a = 1.25
v0 = (k1 / a) * (k1 / a) / 2
e = v0 * r

dt = 0.1

kernelMat = scipy.sparse.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(n, n)).toarray()
pMat = scipy.sparse.spdiags(createBarrier(0, a, v0), 0, n, n).toarray()
# np.array([[0] * n] * n)


# scipy.sparse.spdiags((x / 32.0) ** 2 / 2, 0, n, n).toarray()

fig, ax = plt.subplots()
line1, = plt.plot([], color='red')
line2, = plt.plot([], color='blue')

ax.fill_betweenx([-10, 10], 0, a, color=(1, 0.8, 0.8, 0.6))

plt.xlim(start, end)
if usePDF:
    plt.ylim(0.0, 0.3)
else:
    plt.ylim(-0.6, 0.6)

def hamiltonian():
    ham = pMat - kernelMat / (2 * dx**2)
    return ham
def timeEvoOp(ham):
    u = scipy.linalg.expm(ham * -1j * dt)
    return u
def gwp(x, a, x0, p0, sigma0):
    return a * np.exp(-((x - x0) / (2 * sigma0)) ** 2 + p0 * x * 1j)
def wavePacket(x0, p0, sigma0):
    wf = []
    a = (2 * np.pi * sigma0 * sigma0) ** (-0.25)

    for i in range(0, n):
        wf.append([0])
        wf[i][0] = gwp(i / scale + start, a, x0, p0, sigma0)
    return np.array(wf)
def getPDF(wf):
    return wf.real * wf.real + wf.imag * wf.imag
def step(wf):
    global ham
    teo = timeEvoOp(ham)
    return np.einsum('ij...,jk...->ik...', teo, wf)

ham = hamiltonian()
waveFunction = wavePacket(-48, np.sqrt(2 * e), 3)
# waveFunction = wavePacket(20, 1, 2)

def render(t):
    global waveFunction
    waveFunction = step(waveFunction)

    if (t % 10 == 0):
        print('%s%% finished' % (t / 1000 * 100))

    if usePDF:
        line1.set_data((x, getPDF(waveFunction)))
    else:
        line1.set_data((x, waveFunction.real))
        line2.set_data((x, waveFunction.imag))

ani = FuncAnimation(fig, render, frames=1000, interval=16, repeat=False)
ani.save('wf-undefined.mp4', writer='ffmpeg')

print('%s minutes' % (round(round(time.time() - startTime) / 60 * 100) / 100))
# plt.show()