import networkx as nx
import random
import pandas as pd


def brdmodel(G_orig, p, eta):  # naming lambda "p" to avoid conflict with lambda function
    '''
    Implements the BRD Model, a variant of the SI compartmental model which stochastically sorts 
    exposed nodes into one of two compartments, inspired by belief adoption.

    G_orig (nx.Graph): Network on which to run the process
    p (float): Probability for exposed node to become a Believer instead of a Disbeliever
    eta (float): Constant probability for Disbeliever to turn into a Believer at each timestep
    max (int): Maximum number of allowed iterations

    returns DataFrame: A DataFrame of timesteps and their corresponding compartment counts
    '''
    G = G_orig.copy()

    size = len(set(G.nodes()))
    max_steps = size**2  # allow max iterations to scale with size

    seed = random.choice(list(G.nodes()))
    exposed = [seed]

    believers = [seed]
    receivers = set(G.nodes)
    receivers.remove(seed)
    disbelievers = []

    # history storage
    history = []

    # record time 0
    history.append({
        'Time': 0,
        'Believers': len(believers),
        'Receivers': len(receivers),
        'Disbelievers': len(disbelievers)
    })

    # Store the sets of nodes at each step for the GIF
    snapshots = []

    # Save Time 0 state
    snapshots.append({
        'believers': set(believers),
        'receivers': set(receivers),
        'disbelievers': set(disbelievers)
    })

    t = 1
    while len(believers) < size and t < max_steps:
        new_believers = []
        new_disbelievers = []
        for e in believers:
            for n in G.neighbors(e):
                if n not in exposed:
                    if n in receivers:
                        receivers.remove(n)
                    if random.random() < p:
                        new_believers.append(n)
                    else:
                        new_disbelievers.append(n)
                    exposed.append(n)
        believers.extend(new_believers)
        disbelievers.extend(new_disbelievers)

        for d in list(disbelievers):
            if random.random() < eta:
                disbelievers.remove(d)
                believers.append(d)

        snapshots.append({
            'believers': set(believers),
            'receivers': set(receivers),
            'disbelievers': set(disbelievers)
        })

        history.append({
            'Time': t,
            'Believers': len(believers),
            'Receivers': len(receivers),
            'Disbelievers': len(disbelievers)
        })

        t += 1
        if len(receivers) == 0 and len(disbelievers) == 0:
            print("Full saturation reached. Ending simulation...")
            break
        if t == max_steps:
            print('Maximum timesteps reached. Ending simulation...')

    return pd.DataFrame(history), snapshots
