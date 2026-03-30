import os
import re
import yaml
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator, Locator

# Load the c-based Yaml Loader for faster loading
from yaml import CSafeLoader as Loader
from matplotlib.figure import Figure

ROOT_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../'
# LOG_DIR = os.path.join(ROOT_PATH, 'logs/from_home_pc/no_prime/scenario_1')
LOG_DIR = os.path.join(ROOT_PATH, 'logs/recuv_logs/coop/dfa_game/no_prime/scenario_3')
PLOTS_DIR = os.path.join(ROOT_PATH, 'benchmark_plots/coop_plots/no_prime/')


def save_plot(file_name, plt_handle: plt, fig: Figure):
    """
    Simple method to save figure given the figure handle.
    """
    plt_handle.savefig(file_name, dpi=300, bbox_inches='tight')
    plt_handle.close(fig)


def parse_logs(log_directory):
    data_list = []

    # Regex to capture: (boxes)b_(locs)l_(algorithm)_(game_type).yaml
    # Example: 3b_10l_BDD_dfa_game.yaml
    # for scneario 1 and 2
    # file_pattern = re.compile(r"(\d+)b_(\d+)l_([a-zA-Z]+)_(.*)\.yaml")
    # For scenario 3, we also want to capture the ratio
    file_pattern = re.compile(r"(\d+)b_(\d+)l_(\d+)k_([a-zA-Z]+)_(.*)\.yaml")
    # For scenario 4, we also want to capture the ratio
    # file_pattern = re.compile(r"(\d+)b_(\d+)l_(\d+)f_([a-zA-Z]+)_(.*)\.yaml")

    for filename in os.listdir(log_directory):
        if not filename.endswith(".yaml"):
            continue
            
        match = file_pattern.match(filename)
        if match:
            num_boxes = int(match.group(1))
            num_locs = int(match.group(2))
            # algo_type = match.group(3)
            # game_type = match.group(4)
            # for scenario 3
            algo_type = match.group(4)
            game_type = match.group(5)

            # for scenario 4
            # formula_size = match.group(3)

            # just load 3b yaml files
            # if num_boxes != 3:
            #     continue
            
            print("Processing file:", filename)

            file_path = os.path.join(log_directory, filename)
            
            with open(file_path, 'r') as file:
                try:
                    # Use FullLoader for complex YAML structures
                    # content = yaml.load(file, Loader=yaml.FullLoader)
                    content = yaml.load(file, Loader=Loader)
                    
                    tr_times = []
                    synth_times = []
                    memory_per_run = []

                    # Iterate through "Run 0", "Run 1", etc.
                    for run_id, run_data in content.items():
                        if isinstance(run_data, dict) and 'CompTime' in run_data:
                            comp_time = run_data['CompTime']
                            tr_times.append(comp_time['TR_time'])
                            synth_times.append(comp_time['Synth_time'])
                            memory_per_run.append(run_data['MemoryInUse'])

                    # Calculate Averages
                    if tr_times:
                        data_list.append({
                            'boxes': num_boxes,
                            'locs': num_locs,
                            'algo': algo_type,
                            'game': game_type,
                            # 'formula_size': int(formula_size),
                            'avg_memory': (sum(memory_per_run) / len(memory_per_run))/ (1e6),
                            'ratio' : content['Run 0']['Setup']['ratio'],
                            'preimage_size': content['Run 0']['CompTime']['Preimage_size'],
                            'iterations': content['Run 0']['CompTime']['Iterations'],
                            'avg_tr_time': sum(tr_times) / len(tr_times),
                            'avg_synth_time': sum(synth_times) / len(synth_times),
                            'total_time': (sum(tr_times) + sum(synth_times)) / len(tr_times)
                        })

                except Exception as e:
                    print(f"Error parsing {filename}: {e}")

    return pd.DataFrame(data_list)


def plot_memory(df, target_boxes, target_game, log_plot: bool = False):
    """
     Plots avg memory required for computation vs locs for a specific box count and game type.
    """
    # 1. Filter the data for the specific configuration
    filtered_df = df[(df['boxes'] == target_boxes) & (df['game'] == target_game)].copy()
    
    # 2. Sort by locations to ensure lines connect correctly
    filtered_df = filtered_df.sort_values(by='locs')

    # 3. Set the visual style
    sns.set_context("talk") # Increases font scaling automatically
    sns.set_style("white")  # Clean background
    fig, ax = plt.subplots(figsize=(10, 7))

    palette = {"BDD": "#1f77b4", "ADD": "#d62728", "hybrid": "#2ca02c"}
    dashes = {"BDD": "", "ADD": (4, 1.5), "hybrid": (1, 1)} # Solid, Dashed, Dotted

    # 4. Create the line plot
    # hue='algo' handles the three different colors
    # marker='o' adds points to the lines for clarity
    plot = sns.lineplot(
        data=filtered_df, 
        x='locs', 
        y='memory', 
        hue='algo', 
        style='algo',
        palette=palette,
        dashes=dashes,
        marker='o',
        markersize=10,
        linewidth=2.5,
        ax=ax
    )

    # 5. Apply Log Scale to Y-axis
    if log_plot:
        ax.set_yscale('log')
        # Automatically place ticks at powers of 10 and sensible midpoints
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        # ax.yaxis.set_major_locator(Locator())
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        # ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200]) # Common scale points

    sns.despine() # Remove top/right spines
    ax.grid(True, which='both', linestyle='--', alpha=0.4) # Subtle grid

    # 6. Formatting Labels and Title
    plt.title(f'Memory Required: {target_boxes} Boxes ({target_game.upper()})', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of Locations', fontsize=14, fontweight='semibold')
    plt.ylabel('Avg Memory (MB)', fontsize=14, fontweight='semibold')
    
    # Legend Placement
    plt.legend(title='Algorithm', frameon=False, loc='upper left')
    
    # plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if log_plot:
        save_plot(file_name=PLOTS_DIR + f'memory_{target_boxes}b_{target_game}_log.png', plt_handle=plt, fig=plt.gcf())
    else:
        save_plot(file_name=PLOTS_DIR + f'memory_{target_boxes}b_{target_game}.png', plt_handle=plt, fig=plt.gcf())



def plot_synthesis_time_scn_1(df, target_boxes, target_game, plot_memory: bool= False, log_plot: bool = False, prime: bool = False):
    """
    Plots avg_synth_time vs locs for a specific box count and game type.
    """
    # 1. Filter the data for the specific configuration
    filtered_df = df[(df['boxes'] == target_boxes) & (df['game'] == target_game)].copy()
    
    # 2. Sort by locations to ensure lines connect correctly
    filtered_df = filtered_df.sort_values(by='locs')

    # 3. Set the visual style
    # sns.set_theme(style="whitegrid")
    sns.set_context("talk") # Increases font scaling automatically
    sns.set_style("white")  # Clean background
    fig, ax = plt.subplots(figsize=(10, 7))

    palette = {"BDD": "#1f77b4", "ADD": "#d62728", "hybrid": "#2ca02c"}
    dashes = {"BDD": "", "ADD": (4, 1.5), "hybrid": (1, 1)} # Solid, Dashed, Dotted

    # 4. Create the line plot
    # hue='algo' handles the three different colors
    # marker='o' adds points to the lines for clarity
    if plot_memory:
            plot = sns.lineplot(
                data=filtered_df, 
                x='locs', 
                y='avg_memory', 
                hue='algo', 
                style='algo',
                palette=palette,
                dashes=dashes,
                marker='o',
                markersize=10,
                linewidth=2.5,
                ax=ax
            )
    else:
        plot = sns.lineplot(
            data=filtered_df, 
            x='locs', 
            y='avg_synth_time', 
            hue='algo', 
            style='algo',
            palette=palette,
            dashes=dashes,
            marker='o',
            markersize=10,
            linewidth=2.5,
            ax=ax
        )

    # 5. Apply Log Scale to Y-axis
    if log_plot:
        plot.set_yscale('log')
        # Automatically place ticks at powers of 10 and sensible midpoints
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        # ax.yaxis.set_major_locator(Locator())
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        # ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200]) # Common scale points

    sns.despine() # Remove top/right spines
    ax.grid(True, which='both', linestyle='--', alpha=0.4) # Subtle grid

    # 6. Formatting Labels and Title
    if plot_memory:
        plt.title(f'Memory Required: {target_boxes} Boxes ({target_game.upper()})', 
              fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Number of Locations', fontsize=14, fontweight='semibold')
        plt.ylabel('Avg Memory (MB)', fontsize=14, fontweight='semibold')
    else:
        plt.title(f'Synthesis Complexity: {target_boxes} Boxes ({target_game.upper()})', 
                  fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Number of Locations', fontsize=14, fontweight='semibold')
        plt.ylabel('Avg Synthesis Time (s)', fontsize=14, fontweight='semibold')

    # Legend Placement
    plt.legend(title='Algorithm', frameon=False, loc='upper left')
    
    # plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    prime_str = 'prime' if prime else 'no_prime'

    if plot_memory:
        if log_plot:
            save_plot(file_name=PLOTS_DIR + f'memory_{target_boxes}b_{prime_str}_{target_game}_log.png', plt_handle=plt, fig=plt.gcf())
        else:
            save_plot(file_name=PLOTS_DIR + f'memory_{target_boxes}b_{prime_str}_{target_game}.png', plt_handle=plt, fig=plt.gcf())
    else:
        if log_plot:
            save_plot(file_name=PLOTS_DIR + f'synthesis_time_{target_boxes}b_{prime_str}_{target_game}_log.png', plt_handle=plt, fig=plt.gcf())
        else:
            save_plot(file_name=PLOTS_DIR + f'synthesis_time_{target_boxes}b_{prime_str}_{target_game}.png', plt_handle=plt, fig=plt.gcf())



def plot_synthesis_time_scn_2(df, target_locs, target_game, plot_memory: bool= False, log_plot: bool = False, prime: bool = False):
    """
    Plots avg_synth_time vs boxes for a specific location count and game type.
    """
    # 1. Filter the data for the specific configuration
    filtered_df = df[(df['locs'] == target_locs) & (df['game'] == target_game)].copy()
    
    # 2. Sort by locations to ensure lines connect correctly
    filtered_df = filtered_df.sort_values(by='boxes')

    # 3. Set the visual style
    # sns.set_theme(style="whitegrid")
    sns.set_context("talk") # Increases font scaling automatically
    sns.set_style("white")  # Clean background
    fig, ax = plt.subplots(figsize=(10, 7))

    palette = {"BDD": "#1f77b4", "ADD": "#d62728", "hybrid": "#2ca02c"}
    dashes = {"BDD": "", "ADD": (4, 1.5), "hybrid": (1, 1)} # Solid, Dashed, Dotted

    # 4. Create the line plot
    # hue='algo' handles the three different colors
    # marker='o' adds points to the lines for clarity
    if plot_memory:
        plot = sns.lineplot(
            data=filtered_df, 
            x='boxes', 
            y='avg_memory', 
            hue='algo', 
            style='algo',
            palette=palette,
            dashes=dashes,
            marker='o',
            markersize=10,
            linewidth=2.5,
            ax=ax
        )
    else:
        plot = sns.lineplot(
            data=filtered_df, 
            x='boxes', 
            y='avg_synth_time', 
            hue='algo', 
            style='algo',
            palette=palette,
            dashes=dashes,
            marker='o',
            markersize=10,
            linewidth=2.5,
            ax=ax
        )

    # 5. Apply Log Scale to Y-axis
    if log_plot:
        plot.set_yscale('log')
        # Automatically place ticks at powers of 10 and sensible midpoints
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        # ax.yaxis.set_major_locator(Locator())
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        # ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200]) # Common scale points

    sns.despine() # Remove top/right spines
    ax.grid(True, which='both', linestyle='--', alpha=0.4) # Subtle grid

    # 6. Formatting Labels and Title
    if plot_memory:
        plt.title(f'Memory Required: {target_locs} Boxes ({target_game.upper()})', 
              fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Number of Boxes', fontsize=14, fontweight='semibold')
        plt.ylabel('Avg Memory (MB)', fontsize=14, fontweight='semibold')
    else:
        plt.title(f'Synthesis Complexity: {target_locs} Locs ({target_game.upper()})', 
                  fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Number of Boxes', fontsize=14, fontweight='semibold')
        plt.ylabel('Avg Synthesis Time (s)', fontsize=14, fontweight='semibold')
    
    # Legend Placement
    plt.legend(title='Algorithm', frameon=False, loc='upper left')
    
    # plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    prime_str = 'prime' if prime else 'no_prime'

    if plot_memory:
        if log_plot:
            save_plot(file_name=PLOTS_DIR + f'memory_{target_locs}l_{prime_str}_{target_game}_log.png', plt_handle=plt, fig=plt.gcf())
        else:
            save_plot(file_name=PLOTS_DIR + f'memory_{target_locs}l_{prime_str}_{target_game}.png', plt_handle=plt, fig=plt.gcf())
    else:
        if log_plot:
            save_plot(file_name=PLOTS_DIR + f'synthesis_time_{target_locs}l_{prime_str}_{target_game}_log.png', plt_handle=plt, fig=plt.gcf())
        else:
            save_plot(file_name=PLOTS_DIR + f'synthesis_time_{target_locs}l_{prime_str}_{target_game}.png', plt_handle=plt, fig=plt.gcf())


def plot_synthesis_time_scn_3(df, target_boxes, target_locs, target_game, plot_memory: bool =False, log_plot: bool = False, prime: bool = False):
    """
    Plots avg_synth_time vs boxes for a specific location count and game type.
    """
    # 1. Filter the data for the specific configuration
    filtered_df = df[(df['locs'] == target_locs) & (df['game'] == target_game) & (df['boxes'] == target_boxes)].copy()
    
    # 2. Sort by locations to ensure lines connect correctly
    filtered_df = filtered_df.sort_values(by='ratio')

    # 3. Set the visual style
    # sns.set_theme(style="whitegrid")
    sns.set_context("talk") # Increases font scaling automatically
    sns.set_style("white")  # Clean background
    fig, ax = plt.subplots(figsize=(10, 7))

    palette = {"BDD": "#1f77b4", "ADD": "#d62728", "hybrid": "#2ca02c"}
    dashes = {"BDD": "", "ADD": (4, 1.5), "hybrid": (1, 1)} # Solid, Dashed, Dotted

    # 4. Create the line plot
    # hue='algo' handles the three different colors
    # marker='o' adds points to the lines for clarity
    if plot_memory:
        plot = sns.lineplot(
            data=filtered_df, 
            x='ratio', 
            y='avg_memory', 
            hue='algo', 
            style='algo',
            palette=palette,
            dashes=dashes,
            marker='o',
            markersize=10,
            linewidth=2.5,
            ax=ax
        )
    else:
        plot = sns.lineplot(
            data=filtered_df, 
            x='ratio', 
            y='avg_synth_time', 
            hue='algo', 
            style='algo',
            palette=palette,
            dashes=dashes,
            marker='o',
            markersize=10,
            linewidth=2.5,
            ax=ax
        )

    # 5. Apply Log Scale to Y-axis
    if log_plot:
        plot.set_yscale('log')
        # Automatically place ticks at powers of 10 and sensible midpoints
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        # ax.yaxis.set_major_locator(Locator())
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        # ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200]) # Common scale points

    sns.despine() # Remove top/right spines
    ax.grid(True, which='both', linestyle='--', alpha=0.4) # Subtle grid

    # 6. Formatting Labels and Title
    if plot_memory:
        plt.title(f'Memory Required: {target_locs} Boxes {target_locs} Locs ({target_game.upper()})', 
              fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Ratio', fontsize=14, fontweight='semibold')
        plt.ylabel('Avg Memory (MB)', fontsize=14, fontweight='semibold')
    else:
        plt.title(f'Synthesis Complexity: {target_boxes} Boxes {target_locs} Locs ({target_game.upper()})', 
                fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Ratio', fontsize=14, fontweight='semibold')
        plt.ylabel('Avg Synthesis Time (s)', fontsize=14, fontweight='semibold')
    
    # Legend Placement
    plt.legend(title='Algorithm', frameon=False, loc='upper left')
    
    # plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    prime_str = 'prime' if prime else 'no_prime'

    if plot_memory:
        if log_plot:
            save_plot(file_name=PLOTS_DIR + f'memory_{target_boxes}b_{target_locs}l_{prime_str}_{target_game}_log.png', plt_handle=plt, fig=plt.gcf())
        else:
            save_plot(file_name=PLOTS_DIR + f'memory_{target_boxes}b_{target_locs}l_{prime_str}_{target_game}.png', plt_handle=plt, fig=plt.gcf())
    else:
        if log_plot:
            save_plot(file_name=PLOTS_DIR + f'synthesis_time_{target_boxes}b_{target_locs}l_{prime_str}_{target_game}_log.png', plt_handle=plt, fig=plt.gcf())
        else:
            save_plot(file_name=PLOTS_DIR + f'synthesis_time_{target_boxes}b_{target_locs}l_{prime_str}_{target_game}.png', plt_handle=plt, fig=plt.gcf())



def plot_synthesis_time_scn_4(df, target_boxes, target_locs, target_game, plot_memory: bool =False, log_plot: bool = False, prime: bool = False):
    """
    Plots avg_synth_time vs boxes for a specific location count and game type.
    """
    # 1. Filter the data for the specific configuration
    filtered_df = df[(df['locs'] == target_locs) & (df['game'] == target_game) & (df['boxes'] == target_boxes)].copy()
    
    # 2. Sort by locations to ensure lines connect correctly
    filtered_df = filtered_df.sort_values(by='ratio')

    # 3. Set the visual style
    # sns.set_theme(style="whitegrid")
    sns.set_context("talk") # Increases font scaling automatically
    sns.set_style("white")  # Clean background
    fig, ax = plt.subplots(figsize=(10, 7))

    palette = {"BDD": "#1f77b4", "ADD": "#d62728", "hybrid": "#2ca02c"}
    dashes = {"BDD": "", "ADD": (4, 1.5), "hybrid": (1, 1)} # Solid, Dashed, Dotted

    # 4. Create the line plot
    # hue='algo' handles the three different colors
    # marker='o' adds points to the lines for clarity
    if plot_memory:
        plot = sns.lineplot(
            data=filtered_df, 
            x='formula_size', 
            y='avg_memory', 
            hue='algo', 
            style='algo',
            palette=palette,
            dashes=dashes,
            marker='o',
            markersize=10,
            linewidth=2.5,
            ax=ax
        )
    else:
        plot = sns.lineplot(
            data=filtered_df, 
            x='formula_size', 
            y='avg_synth_time', 
            hue='algo', 
            style='algo',
            palette=palette,
            dashes=dashes,
            marker='o',
            markersize=10,
            linewidth=2.5,
            ax=ax
        )

    # 5. Apply Log Scale to Y-axis
    if log_plot:
        plot.set_yscale('log')
        # Automatically place ticks at powers of 10 and sensible midpoints
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        # ax.yaxis.set_major_locator(Locator())
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        # ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200]) # Common scale points

    sns.despine() # Remove top/right spines
    ax.grid(True, which='both', linestyle='--', alpha=0.4) # Subtle grid

    # 6. Formatting Labels and Title
    if plot_memory:
        plt.title(f'Memory Required: {target_locs} Boxes {target_locs} Locs ({target_game.upper()})', 
              fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Formula Size', fontsize=14, fontweight='semibold')
        plt.ylabel('Avg Memory (MB)', fontsize=14, fontweight='semibold')
    else:
        plt.title(f'Synthesis Complexity: {target_boxes} Boxes {target_locs} Locs ({target_game.upper()})', 
                fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Formula Size', fontsize=14, fontweight='semibold')
        plt.ylabel('Avg Synthesis Time (s)', fontsize=14, fontweight='semibold')
    
    # Legend Placement
    plt.legend(title='Algorithm', frameon=False, loc='upper left')
    
    # plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    prime_str = 'prime' if prime else 'no_prime'

    if plot_memory:
        if log_plot:
            save_plot(file_name=PLOTS_DIR + f'memory_{target_boxes}b_{target_locs}l_{prime_str}_{target_game}_log.png', plt_handle=plt, fig=plt.gcf())
        else:
            save_plot(file_name=PLOTS_DIR + f'memory_{target_boxes}b_{target_locs}l_{prime_str}_{target_game}.png', plt_handle=plt, fig=plt.gcf())
    else:
        if log_plot:
            save_plot(file_name=PLOTS_DIR + f'synthesis_time_{target_boxes}b_{target_locs}l_{prime_str}_{target_game}_formula_log.png', plt_handle=plt, fig=plt.gcf())
        else:
            save_plot(file_name=PLOTS_DIR + f'synthesis_time_{target_boxes}b_{target_locs}l_{prime_str}_{target_game}_formula.png', plt_handle=plt, fig=plt.gcf())


def plot_winning_region_size(df, boxes: int, locs: int, game: str, fig_title: str = ''):
    """
    For givem instance of game, the method Plots winning region size for different algorithms. 
     Spefically, we are plotting the size of the preimages for ADD approach vs BDD. 
     The size of DDs for hybrid and BDD must be eaxctly the same.
    
    We plot a box plot (similar to our TRO paper) to show the distibution of preimage size of BDD across the layer. 
    The size of ADD is singleton and thus we plot it as solid marked overlaid on the box plot
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    filtered_df = df[(df['boxes'] == boxes) & (df['locs'] == locs) & (df['game'] == game)].copy()
    ADDSizesDict = filtered_df[filtered_df['algo'] == 'ADD']['preimage_size'].values[0]
    BDDSizesDict = filtered_df[filtered_df['algo'] == 'BDD']['preimage_size'].values[0]

    assert filtered_df[filtered_df['algo'] == 'ADD']['iterations'].values[0] == filtered_df[filtered_df['algo'] == 'BDD']['iterations'].values[0], \
        "The VI took different number of iterations for ADD and BDD, cannot compare preimage sizes across layers."

    iterations: int = filtered_df[filtered_df['algo'] == 'ADD']['iterations'].values[0]
    ADDSizesList = [ADDSizesDict[i][0] for i in range(iterations)]

    # Enable TeX rendering
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'
    meanprpos = {'marker': 'D',          # Diamond marker
                'markerfacecolor': 'red',  # Red fill
                'markeredgecolor': 'black', # Black outline
                'markersize': 6}

    data_to_plot = []
    for i in range(iterations):
        win_size = []
        for preimage_size, postimage_size in BDDSizesDict[i].values():
            win_size.append(preimage_size)
        data_to_plot.append(np.array(win_size))
    
    # setting up things
    fig, ax = plt.subplots()
    ax.set_ylabel('Size', fontsize=16, labelpad=10)
    
    bplot = plt.boxplot(positions=list(range(iterations)),
                        labels=list(range(iterations)),     ### Overide the labels with the Env labels later.
                        x=data_to_plot,
                        # showmeans=True,
                        # meanprops=meanprpos,
                        patch_artist=True, 
                        showfliers=False)
    ax.scatter(
        range(iterations), 
        ADDSizesList, 
        # **meanprpos,
        color='red', 
        marker='D', 
        s=10, 
        edgecolors='black',
        linewidths=0.5,
        # label='ADD Size', 
        zorder=3  # Ensures it stays on top of the boxes
    )
    
    # min_count = min(sample_counts)
    # max_count = max(sample_counts)
    # norm = mcolors.Normalize(vmin=min_count, vmax=max_count)

    # Create a colormap - using a sequential colormap
    # cmap = plt.cm.viridis  # You can try other colormaps like 'plasma', 'inferno', 'magma', etc.
    # cmap = plt.cm.Greys  # Using the Greys colormap for grayscale
    # cmap = plt.cm.Greys_r  # Using the Inverted Greys colormap for grayscale
    # cmap = plt.cm.coolwarm
    # line_style = '--'  # dashed line
    for i, patch in enumerate(bplot['boxes']):
        patch.set_linestyle('-')
        patch.set_facecolor('white')
        patch.set_edgecolor('black')
        
        # Also color the median, whiskers, caps, and fliers to match
        # ['medians', 'whiskers', 'caps', 'fliers'] - Org list
        for element in ['medians', 'whiskers', 'caps']:
            if element == 'whiskers' or element == 'caps':
                bplot[element][i].set_color('black')
                bplot[element][i].set_linestyle('-')
            elif element in bplot:
                bplot[element][i].set_color('black')  # Keep median line black for contrast
                
                if element == 'fliers':  # Make outlier points darker for visibility
                    bplot[element][i].set_markerfacecolor('black')
                    bplot[element][i].set_markeredgecolor('black')
    
    # ax.set_xticks(range(iterations))
    # ax.set_xticklabels([f'{i}' for i in range(iterations)])
    ax.set_xticks(range(0, iterations, 5))
    ax.set_xticklabels([f'{i}' for i in range(0, iterations, 5)])

    # ax.tick_params(axis='x', which='major', labelsize=12, pad=8)  # Larger x-tick labels
    # ax.tick_params(axis='y', which='major', labelsize=12)

    ax.grid(True, linestyle='--', alpha=0.7, axis='y')  # Only y-axis grid lines
    # Set the grid to appear behind the plot elements
    ax.set_axisbelow(True)

    legend_handles = []
    legend_handles.append(mpatches.Patch(
                            facecolor='white',
                            linestyle='-',
                            edgecolor='black',
                            label='BDD/hybrid'
                        ))
    from matplotlib.lines import Line2D
    legend_handles.append(Line2D( [0], [0], color='none', **meanprpos, label='ADD'))

    ax.legend(handles=legend_handles, 
            # loc='lower right', 
            #   bbox_to_anchor=(0.12, 0.99),
            #   title='System Strategies',
            #   title_fontsize=20,
              fontsize=12)
    # Set title
    if fig_title != '':
        # ax.set_title('Default', fontsize=10)
        plt.title(fig_title)
            
    # Make sure the figure fits well with the colorbar
    plt.tight_layout()
    save_plot(PLOTS_DIR + f'winning_region_size_{boxes}b_{locs}l_{game}.png', plt_handle=plt, fig=fig)
    


if __name__ == "__main__":

    # --- Usage ---
    # log_dir = "./logs"  # Replace with your actual path
    df = parse_logs(LOG_DIR)

    # Display the first few rows to verify
    # print(df.sort_values(by=['boxes', 'locs']))

    # plot a line chart
    # plot_synthesis_time_scn_1(df=df, target_boxes=3, target_game='dfa_game', plot_memory=True, prime=False, log_plot=False)
    # plot_synthesis_time_scn_2(df=df, target_locs=8, target_game='dfa_game', plot_memory=True, prime=False, log_plot=False)
    plot_synthesis_time_scn_3(df=df, target_boxes=4, target_locs=8, target_game='dfa_game', plot_memory=True, prime=False, log_plot=False)
    # plot_synthesis_time_scn_4(df=df, target_boxes=4, target_locs=8, target_game='dfa_game', prime=False, log_plot=False)
    # plot_memory(df=df, target_boxes=3, target_game='dfa_game', log_plot=False)
    # plot_winning_region_size(df=df, boxes=4, locs=15, game='dfa_game')
