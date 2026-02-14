import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import tempfile
import os
import time
from brdmodel import brdmodel
import numpy as np

st.set_page_config(page_title="BRD Model Explorer", layout="wide")

# --- Helper: Create GIF ---


def create_gif(G, pos, snapshots):
    filenames = []
    with tempfile.TemporaryDirectory() as temp_dir:
        # Limit to 60 frames max to keep it fast
        step_size = max(1, len(snapshots) // 60)

        # Progress bar
        progress_text = "Rendering GIF..."
        my_bar = st.progress(0, text=progress_text)

        for i, snapshot in enumerate(snapshots[::step_size]):
            # Create figure
            fig, ax = plt.subplots(figsize=(6, 6))

            # Determine colors
            node_colors = []
            for n in G.nodes():
                if n in snapshot['believers']:
                    node_colors.append('#2ca02c')  # Green
                elif n in snapshot['disbelievers']:
                    node_colors.append('#d62728')  # Red
                else:
                    node_colors.append('#cccccc')  # Gray

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

            # Save frame
            filename = os.path.join(temp_dir, f"frame_{i}.png")
            plt.savefig(filename, dpi=80)
            plt.close(fig)
            filenames.append(filename)

            # Update progress
            percent_complete = (i + 1) / len(snapshots[::step_size])
            my_bar.progress(percent_complete, text=progress_text)

        my_bar.empty()

        # Build GIF
        # We create a persistent file in the temp directory
        gif_path = os.path.join(tempfile.gettempdir(), "network_animation.gif")
        with imageio.get_writer(gif_path, mode='I', duration=0.15, loop=0) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        return gif_path

# Graph generation function


def graph_generator(graph_type, n_nodes):
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
    belief_times = []

    if param_type == 'size':
        param_range = np.arange(10, 1001, 50)
    else:
        param_range = np.linspace(0, 1, num=50)

    for val in param_range:
        test_n = val if param_type == 'size' else current_n
        test_p = val if param_type == 'lambda' else current_p
        test_eta = val if param_type == 'eta' else current_eta

        test_G = graph_generator(graph_type,
                                 n_nodes=test_n) if param_type == 'size' else _current_G

        df, _ = brdmodel(test_G, test_p, test_eta)

        b_df = df[df['Believers'] == test_n]

        if not b_df.empty:
            belief_times.append(b_df['Time'].iloc[0])
        else:
            belief_times.append(None)

    return param_range, belief_times


# --- Sidebar ---
if 'G' not in st.session_state:
    st.session_state.G = None
    st.session_state.pos = None
    st.session_state.simulation_data = None

# --- 2. SIDEBAR INPUTS ---
graph_type = st.sidebar.selectbox("Graph Type", [
                                  "Watts-Strogatz", "Barabasi-Albert", "Erdos-Renyi", "Random Geometric"])
n_nodes = st.sidebar.slider("Number of Nodes", 10, 5000, 100)
p_param = st.sidebar.slider("Adoption Probability (λ)", 0.0, 1.0, 0.5)
eta_param = st.sidebar.slider("Conversion Probability (η)", 0.0, 1.0, 0.05)

# --- 3. GRAPH GENERATION (The "Engine") ---
# We regenerate the graph only if the slider or type changes
if st.session_state.G is None or st.session_state.G.number_of_nodes() != n_nodes:
    st.session_state.G = graph_generator(graph_type, n_nodes)
    # Note: spring_layout can be slow for 5000 nodes; consider a simpler layout for large N
    st.session_state.pos = nx.spring_layout(st.session_state.G, seed=42)

# Alias for easier use in your existing functions
G = st.session_state.G
pos = st.session_state.pos

st.sidebar.markdown("---")
# --- 4. SENSITIVITY BUTTONS (Now safe because G exists) ---
if st.sidebar.button("Analyze Size Sensitivity", key="size_sense",
                     help="Varies size while holding λ and η constant at current slider values"):
    with st.spinner("Running simulations..."):
        # Pass the current values into your function
        # Use defaults or sliders
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

if st.sidebar.button("Clear All Cache"):
    st.cache_data.clear()

# --- 5. RUN SIMULATION LOGIC ---
is_locked = n_nodes > 750
generate_gif = st.sidebar.checkbox("Generate GIF (Slower)",
                                   value=False,
                                   disabled=is_locked,
                                   help="Only available with size <= 750")

# The Primary Button
run_sim = st.sidebar.button("Run Simulation", type="primary")

if run_sim:
    with st.spinner("Running Simulation..."):
        # 1. Run Model (using G and pos from session state)
        df, snapshots = brdmodel(G, p_param, eta_param)

        # 2. Handle GIF
        gif_path = None
        if generate_gif:
            gif_path = create_gif(G, pos, snapshots)

        # 3. Save to Session State so the display section below can see it
        st.session_state['simulation_data'] = {
            'df': df,
            'gif_path': gif_path,
            'final_snapshot': snapshots[-1]
        }

# --- Display Results ---
if 'simulation_data' in st.session_state and st.session_state['simulation_data'] is not None:
    data = st.session_state['simulation_data']
    df = data['df']

    if data['gif_path']:
        col1, col2 = st.columns([1, 1])
    else:
        col1 = st.container()
        col2 = None

    with col1:
        st.subheader("Results")
        st.line_chart(df.set_index("Time")[["Believers", "Disbelievers", "Receivers"]],
                      color=["#2ca02c", "#d62728", "#7f7f7f"])

        # Metrics
        final_b = df.iloc[-1]['Believers']
        final_d = df.iloc[-1]['Disbelievers']
        total = final_b + final_d + df.iloc[-1]['Receivers']

        saturated_df = df[df['Receivers'] == 0]
        fullbelief_df = df[df['Believers'] == total]

        if fullbelief_df.empty:
            if total > 0:
                st.metric("Final Adoption", f"{final_b/total:.1%}")
            else:
                st.metric("Final Adoption", "0.0%")

        else:
            fb_time = fullbelief_df['Time'].iloc[0]
            st.metric("Time to Full Belief", f"{fb_time}")
            if not saturated_df.empty:
                sat_time = saturated_df['Time'].iloc[0]
                st.metric('Saturation Time', f"{sat_time}")
                st.metric('Ratio', f"{(fb_time - sat_time)/fb_time:.1%}")

    if col2:
        with col2:
            st.subheader("Network Evolution")

            # Display the GIF
            # We assume the file exists at the path stored in session state
            st.image(
                data['gif_path'],
            )

            # The Replay Button
            # When clicked, Streamlit reruns the script.
            # Because we read the file again in st.image above, it usually restarts.
            # If it doesn't, we can force a cache bust.
            if st.button("Force Rerun"):
                # This clears the cache for this specific element effectively by forcing a rerun
                st.rerun()

else:
    st.info("👈 Set parameters and click 'Run Simulation' to start.")
