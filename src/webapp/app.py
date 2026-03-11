"""Gradio WebApp for house price prediction with a real estate theme."""

import logging

import gradio as gr
import httpx

from src.config import API_HOST, API_PORT

logger = logging.getLogger(__name__)

# In Docker, API_HOST is set to the service name (e.g. "api").
# Locally, 0.0.0.0 is not routable as a client address; use localhost instead.
_host = "localhost" if API_HOST == "0.0.0.0" else API_HOST
API_URL = f"http://{_host}:{API_PORT}"

# King County ZIP codes (most common ones for defaults/examples)
KC_ZIPCODES = [
    98001,
    98002,
    98003,
    98004,
    98005,
    98006,
    98007,
    98008,
    98010,
    98011,
    98014,
    98019,
    98022,
    98023,
    98024,
    98027,
    98028,
    98029,
    98030,
    98031,
    98032,
    98033,
    98034,
    98038,
    98039,
    98040,
    98042,
    98045,
    98052,
    98053,
    98055,
    98056,
    98058,
    98059,
    98065,
    98070,
    98072,
    98074,
    98075,
    98077,
    98092,
    98102,
    98103,
    98105,
    98106,
    98107,
    98108,
    98109,
    98112,
    98115,
    98116,
    98117,
    98118,
    98119,
    98122,
    98125,
    98126,
    98133,
    98136,
    98144,
    98146,
    98148,
    98155,
    98166,
    98168,
    98177,
    98178,
    98188,
    98198,
    98199,
]


def predict_price(
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    waterfront,
    view,
    condition,
    grade,
    sqft_above,
    sqft_basement,
    yr_built,
    yr_renovated,
    zipcode,
    lat,
    long_,
    sqft_living15,
    sqft_lot15,
    year_sold,
    month_sold,
):
    """Call the FastAPI prediction endpoint."""
    payload = {
        "bedrooms": int(bedrooms),
        "bathrooms": float(bathrooms),
        "sqft_living": int(sqft_living),
        "sqft_lot": int(sqft_lot),
        "floors": float(floors),
        "waterfront": int(waterfront),
        "view": int(view),
        "condition": int(condition),
        "grade": int(grade),
        "sqft_above": int(sqft_above),
        "sqft_basement": int(sqft_basement),
        "yr_built": int(yr_built),
        "yr_renovated": int(yr_renovated),
        "zipcode": int(zipcode),
        "lat": float(lat),
        "long": float(long_),
        "sqft_living15": int(sqft_living15),
        "sqft_lot15": int(sqft_lot15),
        "year_sold": int(year_sold),
        "month_sold": int(month_sold),
    }

    try:
        response = httpx.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        price = result["predicted_price"]
        formatted = result["formatted_price"]

        return (
            f"## Estimated Price: {formatted}\n\n"
            f"*Raw value: ${price:,.2f}*\n\n"
            f"---\n"
            f"King County, Washington"
        )
    except httpx.ConnectError:
        return "**Error:** Cannot connect to the API. Is the server running?"
    except httpx.HTTPStatusError as e:
        return f"**Error:** API returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"**Error:** {e}"


def build_app() -> gr.Blocks:
    """Build the Gradio interface."""

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(theme=theme, title="King County House Price Estimator") as app:
        gr.Markdown(
            """
            # King County House Price Estimator
            ### Predict real estate prices in the Seattle metropolitan area

            Enter the property details below to get an estimated sale price
            powered by our XGBoost model trained on 21,000+ historical transactions.

            ---
            """
        )

        with gr.Row():
            # --- Left column: Property Details ---
            with gr.Column(scale=2):
                gr.Markdown("#### Property Details")

                with gr.Row():
                    bedrooms = gr.Slider(0, 12, value=3, step=1, label="Bedrooms")
                    bathrooms = gr.Slider(0, 8, value=2, step=0.25, label="Bathrooms")
                    floors = gr.Slider(1, 3.5, value=1.5, step=0.5, label="Floors")

                with gr.Row():
                    sqft_living = gr.Number(value=2000, label="Living area (sqft)")
                    sqft_lot = gr.Number(value=5000, label="Lot size (sqft)")

                with gr.Row():
                    sqft_above = gr.Number(value=1500, label="Above ground (sqft)")
                    sqft_basement = gr.Number(value=500, label="Basement (sqft)")

                gr.Markdown("#### Quality & Features")

                with gr.Row():
                    condition = gr.Slider(1, 5, value=3, step=1, label="Condition (1-5)")
                    grade = gr.Slider(1, 13, value=8, step=1, label="Grade (1-13)")

                with gr.Row():
                    waterfront = gr.Radio(
                        choices=[0, 1],
                        value=0,
                        label="Waterfront",
                        info="Does the property face the water?",
                    )
                    view = gr.Slider(0, 4, value=0, step=1, label="View quality (0-4)")

            # --- Right column: Location & History ---
            with gr.Column(scale=2):
                gr.Markdown("#### Construction")

                with gr.Row():
                    yr_built = gr.Number(value=1990, label="Year built")
                    yr_renovated = gr.Number(value=0, label="Year renovated (0 = never)")

                gr.Markdown("#### Location")

                with gr.Row():
                    zipcode = gr.Dropdown(
                        choices=KC_ZIPCODES,
                        value=98103,
                        label="ZIP code",
                        filterable=True,
                    )

                with gr.Row():
                    lat = gr.Number(value=47.6516, label="Latitude")
                    long_ = gr.Number(value=-122.3480, label="Longitude")

                gr.Markdown("#### Neighborhood")

                with gr.Row():
                    sqft_living15 = gr.Number(value=1800, label="Avg neighbor living (sqft)")
                    sqft_lot15 = gr.Number(value=5000, label="Avg neighbor lot (sqft)")

                gr.Markdown("#### Sale Date")

                with gr.Row():
                    year_sold = gr.Number(value=2015, label="Year of sale")
                    month_sold = gr.Slider(1, 12, value=6, step=1, label="Month of sale")

        gr.Markdown("---")

        with gr.Row():
            predict_btn = gr.Button(
                "Estimate Price",
                variant="primary",
                size="lg",
            )

        output = gr.Markdown(
            value="*Enter property details and click 'Estimate Price'*",
            label="Prediction Result",
        )

        predict_btn.click(
            fn=predict_price,
            inputs=[
                bedrooms,
                bathrooms,
                sqft_living,
                sqft_lot,
                floors,
                waterfront,
                view,
                condition,
                grade,
                sqft_above,
                sqft_basement,
                yr_built,
                yr_renovated,
                zipcode,
                lat,
                long_,
                sqft_living15,
                sqft_lot15,
                year_sold,
                month_sold,
            ],
            outputs=output,
        )

        # --- Examples ---
        gr.Markdown("### Example Properties")
        gr.Examples(
            examples=[
                [
                    3,
                    2.5,
                    2000,
                    5000,
                    2.0,
                    0,
                    0,
                    3,
                    8,
                    1500,
                    500,
                    1990,
                    0,
                    98103,
                    47.6516,
                    -122.348,
                    1800,
                    5000,
                    2015,
                    6,
                ],
                [
                    4,
                    3.0,
                    3500,
                    8000,
                    2.0,
                    1,
                    4,
                    5,
                    11,
                    2500,
                    1000,
                    2005,
                    0,
                    98039,
                    47.6305,
                    -122.240,
                    3200,
                    7500,
                    2015,
                    5,
                ],
                [
                    2,
                    1.0,
                    900,
                    3000,
                    1.0,
                    0,
                    0,
                    3,
                    6,
                    900,
                    0,
                    1960,
                    0,
                    98178,
                    47.5005,
                    -122.236,
                    1100,
                    4000,
                    2014,
                    10,
                ],
            ],
            inputs=[
                bedrooms,
                bathrooms,
                sqft_living,
                sqft_lot,
                floors,
                waterfront,
                view,
                condition,
                grade,
                sqft_above,
                sqft_basement,
                yr_built,
                yr_renovated,
                zipcode,
                lat,
                long_,
                sqft_living15,
                sqft_lot15,
                year_sold,
                month_sold,
            ],
            label="Click an example to fill in the form",
        )

        gr.Markdown(
            """
            ---
            *DATA713 MLOps Project | King County House Sales Dataset |
            Model: XGBoost + Optuna | Tracking: MLflow*
            """
        )

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
