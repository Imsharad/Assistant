import requests
import urllib.parse


def generate_chartjs_config(config: str) -> dict:
    """
    Generate a Chart.js configuration.
    """
    return {"config": config}


def get_chart_url_from_config(config: str) -> str:
    """
    Generate a Chart.js chart from a configuration.
    """
    if not config:
        raise ValueError("Config cannot be empty")

    urlencode_config = urllib.parse.quote_plus(config)
    chart_url = f"https://quickchart.io/chart?bkg=white&c={urlencode_config}"

    chart_response = requests.get(chart_url)
    if chart_response.status_code != 200:
        raise requests.HTTPError(f"Failed to generate chart. Status code: {chart_response.status_code}")

    return chart_url


def generate_chartjs_config_and_chart(config: str) -> tuple[dict, str]:
    """
    Generate a Chart.js configuration and chart URL.
    """
    chart_config = generate_chartjs_config(config)
    chart_url = get_chart_url_from_config(config)
    return chart_config, chart_url