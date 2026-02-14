import networkx as nx
import random
import pandas as pd


def brdmodel(G_orig, p, eta):  # naming lambda "p" to avoid conflict with lambda function
    '''
    Implements the BRD Model, a variant of the SI compartmental model which stochastically sorts 
    exposed nodes into one of two compartments, inspired by belief adoption.

    Parameters:
    -----------
    G_orig (nx.Graph): Network on which to run the process
    p (float): Probability for an exposed node to become a Believer instead of a Disbeliever (adoption probability, λ)
    eta (float): Constant probability for a Disbeliever to turn into a Believer at each timestep (conversion probability, η)

    Returns:
    --------
    tuple:
        - pd.DataFrame: A DataFrame mapping timesteps to their corresponding compartment counts (History).
        - list: A list of dictionaries containing the sets of 'believers', 'receivers', and 'disbelievers' at each timestep (Snapshots).
    '''
    # create a copy of the graph to avoid modifying the original input graph
    G = G_orig.copy()

    # determine total nodes to set a dynamic upper bound on iterations
    size = len(set(G.nodes()))
    max_steps = size**2  # allow max iterations to scale with size

    # randomly select initial seed node
    seed = random.choice(list(G.nodes()))
    exposed = [seed]

    # initialize compartments: seed starts as a Believer, everyone else is a Receiver
    believers = [seed]
    receivers = set(G.nodes)
    receivers.remove(seed)
    disbelievers = []

    # history storage for aggregate metrics
    history = []

    # record time 0
    history.append({
        'Time': 0,
        'Believers': len(believers),
        'Receivers': len(receivers),
        'Disbelievers': len(disbelievers)
    })

    # store the sets of nodes at each step for the gif generation in the streamlit app
    snapshots = []

    # save time 0 state
    snapshots.append({
        'believers': set(believers),
        'receivers': set(receivers),
        'disbelievers': set(disbelievers)
    })

    t = 1
    # main loop: runs until all nodes are believers or max limits are hit
    while len(believers) < size and t < max_steps:
        new_believers = []
        new_disbelievers = []

        # spread (believers attempt to convert neighboring receivers)
        for e in believers:
            for n in G.neighbors(e):
                # only target neighbors that haven't been exposed yet
                if n not in exposed:
                    if n in receivers:
                        receivers.remove(n)

                    # determine if neighbor adopts or rejects belief
                    if random.random() < p:
                        new_believers.append(n)
                    else:
                        new_disbelievers.append(n)

                    # mark neighbor as exposed so they aren't processed twice
                    exposed.append(n)

        # apply changes for current timestep
        believers.extend(new_believers)
        disbelievers.extend(new_disbelievers)

        # eta evaluation
        for d in list(disbelievers):
            if random.random() < eta:
                disbelievers.remove(d)
                believers.append(d)

        # record exact node distributions for this timestep
        snapshots.append({
            'believers': set(believers),
            'receivers': set(receivers),
            'disbelievers': set(disbelievers)
        })

        # record aggregate statistics for charting
        history.append({
            'Time': t,
            'Believers': len(believers),
            'Receivers': len(receivers),
            'Disbelievers': len(disbelievers)
        })

        t += 1

        # check termination conditions: if no one is left to convert, stop early
        if len(receivers) == 0 and len(disbelievers) == 0:
            print("Full saturation reached. Ending simulation...")
            break
        if t == max_steps:
            print('Maximum timesteps reached. Ending simulation...')

    return pd.DataFrame(history), snapshots
