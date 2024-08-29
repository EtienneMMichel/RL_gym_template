import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots



def plot_continuous_error_bands(means, stds, nb_episodes_inside_env=None):
    nb_episodes_inside_env = (len(means) if nb_episodes_inside_env is None else nb_episodes_inside_env)
    nb_envs = int(len(means)/nb_episodes_inside_env)
    x = list(range(len(means)))
    std_upper = means + stds
    std_lower = means - stds

    means = means.tolist()
    std_upper = std_upper.tolist()
    std_lower = std_lower.tolist()


    fig = go.Figure([
        go.Scatter(
            x=x,
            y=means,
            line=dict(color='rgb(0,100,80)'),
            mode='lines'
        ),
        go.Scatter(
            x=x+x[::-1], # x, then x reversed
            y=std_upper+std_lower[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
    ])
    if nb_envs > 1:
        for i in range(nb_envs):
            fig.add_vline(x=nb_episodes_inside_env*i, line_width=3, line_dash="dash", line_color="black")

    return fig

def make_summary(infos, mode, nb_episodes_inside_env):
    eval(f"make_{mode}_summary(infos, nb_episodes_inside_env)")

def make_train_summary(infos, nb_episodes_inside_env):
    rewards_array = []
    loss_array = []
    rewards_length_array = []
    
    for seed in infos:
        episode_rewards = []
        episode_loss = []
        rewards_length = []
        for episode_info in infos[seed]:
            episode_rewards.append(float(np.array(episode_info["rewards"]).mean()))
            rewards_length.append(len(episode_info["rewards"]))
            episode_loss.append(episode_info["policy_loss"])
        rewards_array.append(episode_rewards)
        rewards_length_array.append(rewards_length)
        loss_array.append(episode_loss)

    rewards_array = np.array(rewards_array)
    rewards_length_array = np.array(rewards_length_array)
    loss_array = np.array(loss_array)

    fig_rewards = plot_continuous_error_bands(means=np.mean(rewards_array, axis=0), stds=np.std(rewards_array, axis=0), nb_episodes_inside_env=nb_episodes_inside_env)
    fig_rewards.update_layout(
        title="Rewards",
        xaxis_title="episodes",
        yaxis_title="mean episode reward",
    )
    fig_loss = plot_continuous_error_bands(means=np.mean(loss_array, axis=0), stds=np.std(loss_array, axis=0), nb_episodes_inside_env=nb_episodes_inside_env)
    fig_loss.update_layout(
        title="Loss",
        xaxis_title="episodes",
        yaxis_title="mean episode loss",
    )
    fig_rewards.show()
    fig_loss.show()
    # fig = make_subplots(
    # rows=1, cols=2,
    # horizontal_spacing=0.02
    # )

    # for i in fig_rewards.data :
    #     fig.add_trace(i, row=1, col=1)
    #     fig.add_

    # for i in fig_loss.data :    
    #     fig.add_trace(i, row=1, col=2)

    # fig.show()

def make_test_summary(infos, nb_episodes_inside_env):
    rewards_array = []
    loss_array = []
    for seed in infos:
        rewards_array.append(infos[seed][0]["rewards"])
        loss_array.append(infos[seed][0]["policy_loss"])

    rewards_array = np.array(rewards_array)
    loss_array = np.array(loss_array)

    fig_rewards = plot_continuous_error_bands(means=np.mean(rewards_array, axis=0), stds=np.std(rewards_array, axis=0))
    fig_rewards.show()
    mean_loss = np.mean(loss_array)