import pandas as pd
import plotly.express as px
import streamlit as st


def plot_response_3d(data: pd.DataFrame, x: str, y: str, z: str):
    fig = px.scatter_3d(
        data_frame=data,
        x=x,
        y=y,
        z=z,
        color=data["machine_failure"].map({0: "success", 1: "failure"}),
        color_discrete_sequence=["blue", "red"],
        title=(
            "<b>RPM vs. Torque vs. Probability of Success</b><br>"
            f"Temperature Ratio=[{data.temp_ratio.min():.3f} - {data.temp_ratio.max():.3f}]"
        ),
    )
    fig.update_layout(
        legend_title_text="Actual Success/Failure",
        legend_itemsizing="constant",
    )

    idx = data[z].argmax()

    fig.update_traces(
        marker=dict(size=2, opacity=0.3),
    )
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(
                    showarrow=True,
                    x=data[x].iloc[idx],
                    y=data[y].iloc[idx],
                    z=data[z].iloc[idx],
                    text=(
                        "<b>Max (probability of success)</b><br>"
                        f"RPM={data[x].iloc[idx]}<br>"
                        f"Torque={data[y].iloc[idx]}<br>"
                    ),
                    xanchor="left",
                    opacity=0.9,
                    ax=0,
                    ay=-30,
                    arrowcolor="red",
                )
            ]
        ),
    )

    st.plotly_chart(fig)
