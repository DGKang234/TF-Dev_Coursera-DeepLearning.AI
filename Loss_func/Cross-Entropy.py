# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]


# plot of distributions
from matplotlib import pyplot
# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
print('P=%.3f Q=%.3f' % (sum(p), sum(q)))
# plot first distribution
pyplot.subplot(2,1,1)
pyplot.bar(events, p)
# plot second distribution
pyplot.subplot(2,1,2)
pyplot.bar(events, q)
# show the plot
pyplot.show()


# Cross-entropy
import numpy as np
def cross_entropy(p, q):
    return -sum([p[i]*np.log(q[i]) for i in range(len(p))])


# the Cross-entropy of P from Q
CE_pq = cross_entropy(p, q)
print(f'H(P, Q): {round(CE_pq, 4)}')

# the CE of Q from P
CE_qp = cross_entropy(q, p)
print(f'H(Q, P): {round(CE_qp, 4)}')



# the Cross-entropy of P from P
CE_pq = cross_entropy(p, p)
print(f'H(P, P): {round(CE_pq, 4)}')

# the CE of Q from Q
CE_qp = cross_entropy(q, q)
print(f'H(Q, Q): {round(CE_qp, 4)}')
