import perceptron as p

def Invert():
    # INVERT
    not1 = p.Perceptron(invoer=[0], weights=[-1], bias=0, trashhold=-0.5)
    print(not1)
    not2 = p.Perceptron(invoer=[1], weights=[-1], bias=0, trashhold=-0.5)
    print(not2)

def And():
    # AND
    and1 = p.Perceptron(invoer=[0, 0], weights=[0.5, 0.5], bias=0, trashhold=1)
    print(and1)
    and2 = p.Perceptron(invoer=[1, 0], weights=[0.5, 0.5], bias=0, trashhold=1)
    print(and2)
    and3 = p.Perceptron(invoer=[0, 1], weights=[0.5, 0.5], bias=0, trashhold=1)
    print(and3)
    and4 = p.Perceptron(invoer=[1, 1], weights=[0.5, 0.5], bias=0, trashhold=1)
    print(and4)

def Or():
    # OR
    or1 = p.Perceptron(invoer=[0, 0], weights=[0.5, 0.5], bias=0, trashhold=0.5)
    print(or1)
    or2 = p.Perceptron(invoer=[1, 0], weights=[0.5, 0.5], bias=0, trashhold=0.5)
    print(or2)
    or3 = p.Perceptron(invoer=[0, 1], weights=[0.5, 0.5], bias=0, trashhold=0.5)
    print(or3)
    or4 = p.Perceptron(invoer=[1, 1], weights=[0.5, 0.5], bias=0, trashhold=0.5)
    print(or4)

def Nor():
    # NOR
    Nor1 = p.Perceptron(invoer=[0, 0, 0], weights=[-1, -1, -1], bias=0, trashhold=0)
    print(Nor1)
    Nor2 = p.Perceptron(invoer=[0, 0, 1], weights=[-1, -1, -1], bias=0, trashhold=0)
    print(Nor2)
    Nor3 = p.Perceptron(invoer=[0, 1, 0], weights=[-1, -1, -1], bias=0, trashhold=0)
    print(Nor3)
    Nor4 = p.Perceptron(invoer=[0, 1, 1], weights=[-1, -1, -1], bias=0, trashhold=0)
    print(Nor4)
    Nor5 = p.Perceptron(invoer=[1, 0, 0], weights=[-1, -1, -1], bias=0, trashhold=0)
    print(Nor5)
    Nor6 = p.Perceptron(invoer=[1, 0, 1], weights=[-1, -1, -1], bias=0, trashhold=0)
    print(Nor6)
    Nor7 = p.Perceptron(invoer=[1, 1, 0], weights=[-1, -1, -1], bias=0, trashhold=0)
    print(Nor7)
    Nor8 = p.Perceptron(invoer=[1, 1, 1], weights=[-1, -1, -1], bias=0, trashhold=0)
    print(Nor8)

def party():
    # party uit de reader ookwel figuur 2.8
    party1 = p.Perceptron(invoer=[0, 0, 0], weights=[0.6, 0.3, 0.2], bias=0, trashhold=0.4)
    print(party1)
    party2 = p.Perceptron(invoer=[0, 0, 1], weights=[0.6, 0.3, 0.2], bias=0, trashhold=0.4)
    print(party2)
    party3 = p.Perceptron(invoer=[0, 1, 0], weights=[0.6, 0.3, 0.2], bias=0, trashhold=0.4)
    print(party3)
    party4 = p.Perceptron(invoer=[0, 1, 1], weights=[0.6, 0.3, 0.2], bias=0, trashhold=0.4)
    print(party4)
    party5 = p.Perceptron(invoer=[1, 0, 0], weights=[0.6, 0.3, 0.2], bias=0, trashhold=0.4)
    print(party5)
    party6 = p.Perceptron(invoer=[1, 0, 1], weights=[0.6, 0.3, 0.2], bias=0, trashhold=0.4)
    print(party6)
    party7 = p.Perceptron(invoer=[1, 1, 0], weights=[0.6, 0.3, 0.2], bias=0, trashhold=0.4)
    print(party7)
    party8 = p.Perceptron(invoer=[1, 1, 1], weights=[0.6, 0.3, 0.2], bias=0, trashhold=0.4)
    print(party8)

# Invert()
# And()
# Or()
# Nor()
party()