def g1():
    for i in range(10):
        yield i

def g2(gen):
    for i in gen:
        for j in range(2):
            yield i, j

gl = list(g1())
print(gl, len(gl))
generater = g2(g1())
gl2 = list(generater)
print(gl2, len(gl2))
