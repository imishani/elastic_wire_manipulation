import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, ode
from mpl_toolkits import mplot3d # Package for 3D plotting
import time
from scipy.spatial import distance

'''
Author: Avishai Sintov
'''
a_global = np.array([3.9632, 3.0471, 4.4825, -22.019, -0.9381, 28.248])

class rod():
    t_solve = 0
    t_collision = 0
    n_solve = 0

    def __init__(self, c=[0.77, 1., 1.], L=1, check_collision=False, check_stable=False, PointsOnRod=500):
        self.c = np.array(c)
        self.L = L
        self.PointsOnRod = PointsOnRod
        self.check_collision = check_collision
        self.check_stable = check_stable
        self.q0 = np.eye(4)
        self.M0=np.eye(6)
        self.J0=np.zeros((6,6))

    def solve(self, a):
        stable, collision = True, False
        self.n_solve += 1
        st = time.time()
        
        x0 = np.concatenate((self.q0[:3,:].reshape(1,-1), a.reshape(1,-1), self.M0.reshape(1,-1), self.J0.reshape(1,-1)), axis=1).reshape((-1,))
        t = np.linspace(0, self.L, self.PointsOnRod)

        # X = odeint(self.eqs, x0, t)#, args=(u,))

        # Much similar to ode45
        X = np.zeros((len(t), len(x0)))   # array for solution
        X[0, :] = x0
        r = ode(self.eqs).set_integrator("dopri5")  # choice of method
        r.set_initial_value(x0, t[0])   # initial values
        for i in range(1, t.size):
            X[i, :] = r.integrate(t[i]) # get one more value, add it to the array
            if not r.successful():
                raise RuntimeError("Could not integrate")

        Q = X[:,:12]
        Q = np.array([np.append(q.reshape((3,4)), np.array([0, 0, 0, 1]).reshape(1,-1), axis=0) for q in Q])

        if self.check_stable:

            J = X[:,54:]
            J = np.array([j.reshape(6,6) for j in J])

            detJ = np.array([np.linalg.det(j) for j in J])
            # print(detJ[:10])
            # print(np.where(detJ < 0), detJ[np.argwhere(detJ < 0)])
            stable = np.argwhere(detJ < 0)
            stable = True if not len(stable) or stable[0] >= self.PointsOnRod-1 else False
        self.t_solve += time.time() - st
        P = Q[:,:3,3]
        st = time.time()
        if self.check_collision:
            collision = self.collision(P[::5,:])
            self.t_collision += time.time() - st

        return Q, stable, collision

    def collision(self, P):
        n = P.shape[0]
        dis = distance.cdist(P, P, 'euclidean')
        r = np.min(np.diag(dis, k=-1))
        dis += np.eye(n) + np.diag(np.ones(n-1), k=-1) + np.diag(np.ones(n-1), k=1)
        dis = np.min(dis, axis=0)

        c = np.argwhere(dis <= r)
        if len(c) > 0:
            return True
        return False

    def eqs(self, t, x):

        q = x[:12].reshape((3,4))
        q = np.append(q, np.array([0, 0, 0, 1]).reshape(1,-1), axis=0)
        mu = x[12:18]
        M = x[18:54].reshape(6,6)
        J = x[54:].reshape(6,6)

        k = 1.0 / self.c
        u = np.multiply(mu[:3], k)

        U = np.array([[0, -u[2], u[1], 1],[u[2], 0, -u[0], 0],[-u[1], u[0], 0, 0,],[0, 0, 0, 0]])
        dq = q.dot(U)

        dmu = np.array([u[2]*mu[1]-u[1]*mu[2],
                mu[5]+u[0]*mu[2]-u[2]*mu[0],
                -mu[4]+u[1]*mu[0]-u[0]*mu[1],
                u[2]*mu[4]-u[1]*mu[5],
                u[0]*mu[5]-u[2]*mu[3],
                u[1]*mu[3]-u[0]*mu[4]])

        F = np.array([[0, mu[2]*(k[2]-k[1]), mu[1]*(k[2]-k[1]), 0,    0,     0],
            [mu[2]*(k[0]-k[2]), 0,  mu[0]*(k[0]-k[2]), 0,     0,     1],
            [mu[1]*(k[1]-k[0]), mu[0]*(k[1]-k[0]),  0,  0,     -1,    0],
            [0,     -mu[5]*k[1],        mu[4]*k[2],        0,     u[2],  -u[1]],
            [mu[5]*k[0],        0,    -mu[3]*k[2],       -u[2], 0,     u[0]],
            [-mu[4]*k[0],       mu[3]*k[1],         0,    u[1],  -u[0], 0]])

        G = np.array([[k[0], 0,    0,    0, 0, 0],
            [0,    k[1], 0,    0, 0, 0],
            [0,    0,    k[2], 0, 0, 0],
            [0,    0,    0,    0, 0, 0],
            [0,    0,    0,    0, 0, 0],
            [0,    0,    0,    0, 0, 0]])

        H = np.array([[0,     u[2],  -u[1], 0,     0,     0],
            [-u[2], 0,     u[0],  0,     0,     0],
            [u[1],  -u[0], 0,     0,     0,     0],
            [0,     0,     0,     0,     u[2],  -u[1]],
            [0,     0,     1,     -u[2], 0,     u[0]],
            [0,     -1,    0,     u[1],  -u[0], 0]])

        dM = np.matmul(F, M)
        dJ= np.add(np.matmul(G, M), np.matmul(H, J))

        dx = np.concatenate((dq[:3,:].reshape(1,-1), dmu.reshape(1,-1), dM.reshape(1,-1), dJ.reshape(1,-1)), axis=1).reshape((-1,))

        return dx

    def get_time_stats(self):
        print("Solve time: %f, collision time: %f"%(self.t_solve/self.n_solve, self.t_collision/self.n_solve))

    def gen_random_rod(self, feasible = False):
        if np.random.random() < 0.3:
            a = np.random.random((6,)) * np.array([40, 40, 40, 80, 80, 80]) - np.array([20, 20, 20, 40, 40, 40])
        else:
            a = np.random.random((6,)) * np.array([20, 20, 20, 40, 40, 40]) - np.array([10, 10, 10, 20, 20, 20])

        if feasible:
            while 1:
                _, stable, collision = self.solve(a)
                if stable and not collision:
                    return a
                a = np.random.random((6,)) * np.array([40, 40, 40, 80, 80, 80]) - np.array([20, 20, 20, 40, 40, 40])
        return a

    def plot(self, a = None, Q = None):
        if a is None and Q is None:
            return

        if Q is None:
            Q, _, _ = self.solve(a)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(Q[:,0,3], Q[:,1,3], Q[:,2,3], '-k')
        plt.show()


def get_rod_points(a=a_global, plot=False, stabillity_check=False, collision_check=False, c=[0.77, 0.82, 0.82], L=0.88,
                   rod_points=500):
    R = rod(L=L, check_stable=stabillity_check, check_collision=collision_check, c=c, PointsOnRod=rod_points)

    # a = np.array([ 1.78864442107052,	4.41934440879383,	1.31657801998917,	30.9465268811736,	3.14934280879430,	69.1186473769193])
    # a = R.gen_random_rod(feasible=False)

    Q, stable, collision = R.solve(a)

    if not stable:
        print("Rod is not stable!")
    if collision:
        print("Rod is in self collision!")

    #R.get_time_stats()
    if plot:
        R.plot(a = a, Q = Q)
    return Q
