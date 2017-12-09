import numpy as np

class ParticleFilter_MOMDP(object):
    def __init__(self, momdp):
        self.momdp = momdp
    
    def ParticleFilter(self, particles, action, prev_state, state):
        weight = np.zeros_like(particles, dtype=np.float)
        for y in range(particles.shape[0]):
            for particle in range(particles[y]):
                weight[y] += self.momdp.tx[y, action, prev_state, state]
        weight /= np.sum(weight)
#         new_particles = np.random.choice(np.arange(particles.shape[0]), p=weight, size=np.sum(particles))
#         return np.array([sum(new_particles == y) for y in range(particles.shape[0])])
        return [int(w * np.sum(particles)) for w in weight]

    def OptimalPolicy(self, startParticles, startState, T=3):
        print startState, startParticles
        reach = {t:set() for t in range(T)}
        reach[0].add((startState, tuple(startParticles)))
        for t in range(1, T):
            for s, particles in reach[t - 1]:
                for a in range(self.momdp.a):
                    for n_s in self.momdp.valid_nx_x[a][s]:
                        reach[t].add((n_s, tuple(self.ParticleFilter(np.array(particles), a, s, n_s))))
        for k, v in reach.items():
            print k, v
