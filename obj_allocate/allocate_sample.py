import pulp
import numpy as np

map_matrix = np.array([[100, 2, 100, 10, 100],
                       [2, 100, 3, 100, 100],
                       [100, 3, 100, 100, 10],
                       [10, 100, 100, 100, 5],
                       [100, 100, 10, 5, 100]])

problem = pulp.LpProblem('sample', pulp.LpMinimize)
edge = pulp.LpVariable.dicts('EDGE', (range(5), range(5)), 0, 1, 'Integer')

problem += pulp.lpSum([pulp.lpDot([edge[i][j] for j in range(len(edge[i]))], map_matrix[i])
                       for i in range(len(edge))])
for j in range(len(edge)):
    problem += pulp.lpSum([edge[i][j] for i in range(len(edge))]) <= 1
for i in range(len(edge)):
    problem += pulp.lpSum([edge[i][j] for j in range(len(edge))]) <= 1
for i in range(len(edge)):
    problem += pulp.lpSum([edge[i][j] for j in range(len(edge))]) +\
               pulp.lpSum([edge[j][i] for j in range(len(edge))]) >= 1
for i in range(len(edge)):
    for j in range(i+1, len(edge)):
        problem += edge[i][j] + edge[j][i] <= 1

problem += pulp.lpSum([pulp.lpSum([edge[i][j] for j in range(len(edge[i]))]) for i in range(len(edge))]) == 4

# a = pulp.LpVariable('a', 0, 1)
# b = pulp.LpVariable('b', 0, 1)
#
# problem += a + b
#
# problem += a >= 0
# problem += b >= 0.1
# problem += a + b == 0.5

status = problem.solve()
print "Status", pulp.LpStatus[status]

# print problem

print "Result"
for _k, v in sorted(edge.viewitems()):
    print [v2.value() for _k2, v2 in sorted(v.viewitems())]
# print edge.value()
