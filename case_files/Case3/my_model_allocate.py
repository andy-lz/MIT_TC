def allocate(simple_args, medium_args, hard_args):
    """
    Implement your allocation function here
    You should return a tuple (a1, a2, a3), where
        a1 is the quantity of stock simple you wish to purchase and so forth
    You will buy the stocks at the current price
    
    The argument format is as follows:
        simple_args will be a tuple of (current_price, current_x1, current_x2)
        medium_args will be a tuple of (current_price, current_x1, current_x2, current_x3)
        hard_args will be a tuple of (current_price_history, current_x1, current_x2, current_x3)
            where the current_price_history is the previous 50 prices
            and the current price is the last element of current_price_history

    Note that although we notate for example feature x1 for all the stocks, the 
        features for each stock are unrelated (x1 for simple has no relation to x1 for medium, etc)

    Make sure the total money you allocate satisfies
        (a1 * simple_current_price + a2 * medium_current_price + a3 * hard_current_price) < 100000000
    Quantities may be decimals so don't worry about rounding
    To be safe, you should make sure you're lower than 100000000 by a threshold
    You can check your code with the provided helper test_allocate.py

    Test your allocation function on the provided test set by running python test_allocate.py
    Generate your final submission on the real data set by running python run_allocate.py
    """
    # Sample: retrieve prices and get predictions from models
    simple_price = simple_args[0]
    medium_price = medium_args[0]
    hard_price = hard_args[0][-1]
    simple_prediction = simple_model.predict(*simple_args)
    medium_prediction = medium_model.predict(*medium_args)
    hard_prediction = hard_model.predict(*hard_args)
    P = np.array([simple_price, medium_price, hard_price])
    dP = np.array([simple_prediction - simple_price, medium_prediction - medium_price, hard_prediction - hard_price])
    return list(allocation_logic_3(dP, P, 0.0002))
    # Sample: allocate all money (except a small threshold) to medium
    # return (0, (100000000 - 1) / medium_price, 0)

def f_i(N_i, a):
    if N_i < 0:
        return 0
    return (2/a)*(math.sqrt(a*N_i + 1) - 1)

def f(N, a):
    return (2/a)*(math.sqrt(a*N[0] + 1) - 1) + (2/a)*(math.sqrt(a*N[1] + 1) - 1) + (2/a)*(math.sqrt(a*N[2] + 1) - 1)

def f_der(N_i, a):
    return 1/math.sqrt(a* N_i + 1)

def neg_f(N, a):
    return -1 * f(N,a)

def neg_f_der(N_i,a):
    return -1 * f_der(N_i,a)

def neg_f_ders(N, a):
    return np.array([neg_f_der(N[0], a), neg_f_der(N[1], a), neg_f_der(N[2], a)])

def max_func(N, dP, a):
    return f_i(N[0], a) * dP[0] + f_i(N[1], a) * dP[1] + f_i(N[2], a) * dP[2]

def min_func(N, dP, a):
    return -1 * max_func(N, dP, a)

def constraint0(N, P):
    cons = 0
    for i in range(len(N)):
        cons += N[i] * P[i]
    return cons - 100000000

def constraint1(N, P):
    return -1 * constraint0(N,P)

def allocation_logic_3(dP, P, a):
    for i in range(len(dP)):
        if dP[i] <= 0:
            max_1 = 0
            max_j = -1
            for j in range(len(dP)):
                ev = dP[j] * 100000000.0/P[j] 
                if ev > max_1:
                    max_1 = ev
                    max_j = j
                if ev >0 and ev == max_1:
                    temp = np.array([50000000/P[0], 50000000/P[1], 50000000/P[2]])
                    temp[i] = 0
                    return temp
            if max_j == -1:
                return([0,0,0])
            temp = np.array([0,0,0])
            temp[max_j] = 100000000.0/P[max_j]
            return temp
     #       dP = np.array([dP[i-1], dP[i-2]])
     #       P = np.array([P[i-1], P[i-2]])
    #        val = allocation_logic(dP, P, a)
   #         print(val.x, i)
   #         return val.x, i
    cons = ({'type' : 'ineq', 'fun': lambda N: np.array([constraint1(N,P)]), 'jac': lambda N: np.array([-1 * P[0], -1 * P[1], -1*P[2]])},
            {'type': 'ineq',
            'fun': lambda P: np.array([P[0]]),
            'jac': lambda P: np.array([1,0,0])},
            {'type': 'ineq',
            'fun': lambda P: np.array([P[1]]),
            'jac': lambda P: np.array([0,1,0])},
            {'type': 'ineq',
            'fun': lambda P: np.array([P[2]]),
            'jac': lambda P: np.array([0,0,1])})
    x0 = (np.array([33333333.33/P[0], 33333333.33/P[1], 33333333.33/P[2]]))
    res = scipy.optimize.minimize(min_func, x0, args=(dP, a), constraints=cons, method='SLSQP', options={'disp': True})
    return res.x