import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import tempfile
import os
import time
from brdmodel import brdmodel
import numpy as np

# configure main page settings
st.set_page_config(page_title="BRD Model Explorer", layout="wide")

# --- Helper: Create GIF ---


def create_gif(G, pos, snapshots):
    """
    Generates an animated GIF visualizing the spread of the belief across the network over time.

    Parameters:
    -----------
    G (nx.Graph): The network graph used in the simulation.
    pos (dict): Node layout positions for plotting coordinates.
    snapshots (list): List of state dictionaries tracking compartments at each timestep.

    Returns:
    --------
    str: The local file path to the generated GIF.
    """
    filenames = []
    # use a temporary directory to store individual frame images before compiling
    with tempfile.TemporaryDirectory() as temp_dir:
        # calculate step size to cap frames at 60 to prevent massive rendering times
        step_size = max(1, len(snapshots) // 60)

        progress_text = "Rendering GIF..."
        my_bar = st.progress(0, text=progress_text)

        # render each chosen frame
        for i, snapshot in enumerate(snapshots[::step_size]):
            fig, ax = plt.subplots(figsize=(6, 6))

            # Mmp compartment states to visual colors
            node_colors = []
            for n in G.nodes():
                if n in snapshot['believers']:
                    node_colors.append('green')
                elif n in snapshot['disbelievers']:
                    node_colors.append('red')
                else:
                    node_colors.append('gray')

            # draw graph
            nx.draw(
                G, pos,
                node_color=node_colors,
                with_labels=False,
                node_size=80,
                edge_color="#e0e0e0",
                width=0.5,
                ax=ax
            )
            ax.set_title(f"Timestep: {i * step_size}")

            # save frame locally
            filename = os.path.join(temp_dir, f"frame_{i}.png")
            plt.savefig(filename, dpi=80)
            plt.close(fig)
            filenames.append(filename)

            # update progress bar
            percent_complete = (i + 1) / len(snapshots[::step_size])
            my_bar.progress(percent_complete, text=progress_text)

        my_bar.empty()

        # compile saved frames into a continuous gif using imageio
        gif_path = os.path.join(tempfile.gettempdir(), "network_animation.gif")
        with imageio.get_writer(gif_path, mode='I', duration=0.15, loop=0) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        return gif_path

# Helper: Graph Generation


def graph_generator(graph_type, n_nodes):
    """
    Constructs a NetworkX graph based on the user's selected topology.

    Parameters:
    -----------
    graph_type (str): The name of the desired network topology (e.g., "Watts-Strogatz").
    n_nodes (int): The number of nodes to generate in the graph.

    Returns:
    --------
    nx.Graph: The generated network topology.
    """
    if graph_type == "Watts-Strogatz":
        G = nx.watts_strogatz_graph(n_nodes, 4, 0.1)
        pos = nx.spring_layout(G, seed=42)
    elif graph_type == "Barabasi-Albert":
        G = nx.barabasi_albert_graph(n_nodes, 2)
        pos = nx.spring_layout(G, seed=42)
    elif graph_type == "Erdos-Renyi":
        G = nx.erdos_renyi_graph(n_nodes, 0.1)
        pos = nx.kamada_kawai_layout(G)
    elif graph_type == 'Random Geometric':
        G = nx.random_geometric_graph(n_nodes, 0.15)
        pos = nx.get_node_attributes(G, 'pos')
    return G

# --- Helper: Sensitivity Analysis


@st.cache_data(show_spinner=False)
def sensitivity_analysis(param_type, _current_G, current_n, current_p, current_eta):
    """
    Runs repeated simulations to track how changes to a single parameter 
    affect the time it takes for 100% belief adoption.

    Parameters:
    -----------
    param_type (str): The parameter to sweep ('size', 'lambda', or 'eta').
    _current_G (nx.Graph): The cached graph to use (if not sweeping size).
    current_n (int): Base number of nodes.
    current_p (float): Base adoption probability.
    current_eta (float): Base conversion probability.

    Returns:
    --------
    tuple:
        - np.ndarray: Array representing the range of the swept parameter.
        - list: Corresponding adoption times for each parameter value.
    """
    belief_times = []

    # define sweep range depending on parameter being analyzed
    if param_type == 'size':
        param_range = np.arange(10, 1001, 50)
    else:
        param_range = np.linspace(0, 1, num=50)

    # execute simulation for each step in range
    for val in param_range:
        test_n = val if param_type == 'size' else current_n
        test_p = val if param_type == 'lambda' else current_p
        test_eta = val if param_type == 'eta' else current_eta

        # generate new graph only if we are sweeping network size
        test_G = graph_generator(graph_type,
                                 n_nodes=test_n) if param_type == 'size' else _current_G

        df, _ = brdmodel(test_G, test_p, test_eta)

        # look for the first row where everyone is a believer
        b_df = df[df['Believers'] == test_n]

        if not b_df.empty:
            belief_times.append(b_df['Time'].iloc[0])
        else:
            # max iterations reached without 100% saturation
            belief_times.append(None)

    return param_range, belief_times


# --- Sidebar ---
# init session state variables to track persistent data
if 'G' not in st.session_state:
    st.session_state.G = None
    st.session_state.pos = None
    st.session_state.simulation_data = None

# parameter inputs
graph_type = st.sidebar.selectbox("Graph Type", [
                                  "Watts-Strogatz", "Barabasi-Albert", "Erdos-Renyi", "Random Geometric"])
n_nodes = st.sidebar.slider("Number of Nodes", 10, 2500, 100)
p_param = st.sidebar.slider("Adoption Probability (λ)", 0.0, 1.0, 0.5)
eta_param = st.sidebar.slider("Conversion Probability (η)", 0.0, 1.0, 0.05)

# detect if graph settings changed and regenerate if needed
if st.session_state.G is None or st.session_state.G.number_of_nodes() != n_nodes:
    st.session_state.G = graph_generator(graph_type, n_nodes)
    st.session_state.pos = nx.spring_layout(st.session_state.G, seed=42)

G = st.session_state.G
pos = st.session_state.pos

st.sidebar.markdown("---")

# buttons for sensitivity analysis
if st.sidebar.button("Analyze Size Sensitivity", key="size_sense",
                     help="Varies size while holding λ and η constant at current slider values"):
    with st.spinner("Running simulations..."):
        x, y = sensitivity_analysis('size', G, n_nodes, p_param, eta_param)
        st.write("### Belief Time vs. Size)")
        st.line_chart(dict(zip(x, y)))

if st.sidebar.button("Analyze Lambda Sensitivity", key="lambda_sense",
                     help="Varies λ while holding size and η constant at current slider values"):
    with st.spinner("Analyzing Lambda..."):
        x, y = sensitivity_analysis('lambda', G, n_nodes, p_param, eta_param)
        st.write('### Belief Time vs. Adoption Probability')
        st.line_chart(dict(zip(x, y)))

if st.sidebar.button("Analyze Eta Sensitivity", key="eta_sense",
                     help="Varies η while holding size and λ constant at current slider values"):
    with st.spinner("Analyzing Lambda..."):
        x, y = sensitivity_analysis('eta', G, n_nodes, p_param, eta_param)
        st.write('Belief Time vs. Conversion Probability')
        st.line_chart(dict(zip(x, y)))

# cache clearing button
if st.sidebar.button("Clear All Cache"):
    st.cache_data.clear()

# prevent gif generation on large networks
is_locked = n_nodes > 750
generate_gif = st.sidebar.checkbox("Generate GIF (Slower)",
                                   value=False,
                                   disabled=is_locked,
                                   help="Only available with size <= 750")

run_sim = st.sidebar.button("Run Simulation", type="primary")

# execute model and store outputs into session state
if run_sim:
    with st.spinner("Running Simulation..."):
        df, snapshots = brdmodel(G, p_param, eta_param)

        gif_path = None
        if generate_gif:
            gif_path = create_gif(G, pos, snapshots)

        st.session_state['simulation_data'] = {
            'df': df,
            'gif_path': gif_path,
            'final_snapshot': snapshots[-1]
        }

# --- Display Results ---
# only display visual components if simulation data exists in session state
if 'simulation_data' in st.session_state and st.session_state['simulation_data'] is not None:
    data = st.session_state['simulation_data']
    df = data['df']

    # handle layout changes dynamically based on whether a gif was generated
    if data['gif_path']:
        col1, col2 = st.columns([1, 1])
    else:
        col1 = st.container()
        col2 = None

    with col1:
        st.subheader("Results")
        # plot compartment sizes over time
        st.line_chart(df.set_index("Time")[["Believers", "Disbelievers", "Receivers"]],
                      color=["#2ca02c", "#d62728", "#7f7f7f"])

        # metrics extraction
        final_b = df.iloc[-1]['Believers']
        final_d = df.iloc[-1]['Disbelievers']
        total = final_b + final_d + df.iloc[-1]['Receivers']

        # determine points of key milestones
        saturated_df = df[df['Receivers'] == 0]
        fullbelief_df = df[df['Believers'] == total]

        # display metrics depending on scenario
        if fullbelief_df.empty:
            if total > 0:
                st.metric("Final Adoption", f"{final_b/total:.1%}")
            else:
                st.metric("Final Adoption", "0.0%")

        else:
            fb_time = fullbelief_df['Time'].iloc[0]
            st.metric("Belief Time", f"{fb_time}")
            if not saturated_df.empty:
                sat_time = saturated_df['Time'].iloc[0]
                st.metric('Saturation Time', f"{sat_time}")
                st.metric('Ratio', f"{sat_time/fb_time:.1%}")

    if col2:  # display gif
        with col2:
            st.subheader("Network Evolution")

            st.image(
                data['gif_path'],
            )

            if st.button("Force Rerun"):
                st.rerun()

else:
    # prompt for when app first loads
    st.info("👈 Set parameters and click 'Run Simulation' to start.")
