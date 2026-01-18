import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
import base64
import paho.mqtt.publish as publish
from config.mqtt_config import mqtt_host, mqtt_port, mqtt_user, mqtt_password, mqtt_base_topic
from utils.expense_analysis_utils import get_current_month_expenses, normalize_expenses, plot_expenses_vs_budget

def publish_dataframe(df: pd.DataFrame, topic_suffix: str) -> None:
    payload = df.to_json(orient="split")
    publish.single(
        f"{mqtt_base_topic}/{topic_suffix}",
        payload,
        hostname=mqtt_host,
        port=mqtt_port,
        auth={
            "username": mqtt_user,
            "password": mqtt_password,
        },
    )

def publish_plot(fig: plt.Figure, topic_suffix: str) -> None:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload = base64.b64encode(buf.getvalue()).decode("utf-8")

    publish.single(
        f"{mqtt_base_topic}/{topic_suffix}",
        payload,
        hostname=mqtt_host,
        port=mqtt_port,
        auth={
            "username": mqtt_user,
            "password": mqtt_password,
        },
    )
