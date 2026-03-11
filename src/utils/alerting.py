"""Email alerting utilities using Gmail SMTP."""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.config import (
    ALERT_RECIPIENTS,
    MLFLOW_TRACKING_URI,
    SMTP_EMAIL,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_SERVER,
)

logger = logging.getLogger(__name__)


def send_alert_email(
    subject: str,
    body: str,
    recipients: list[str] | None = None,
    html: bool = False,
) -> bool:
    """Send an alert email via Gmail SMTP.

    Args:
        subject: Email subject.
        body: Email body (plain text or HTML).
        recipients: List of recipient emails. Defaults to ALERT_RECIPIENTS.
        html: Whether body is HTML.

    Returns:
        True if sent successfully.
    """
    recipients = recipients or ALERT_RECIPIENTS

    if not SMTP_EMAIL or not SMTP_PASSWORD:
        logger.warning("SMTP credentials not configured. Skipping email alert.")
        return False

    if not recipients:
        logger.warning("No alert recipients configured. Skipping email alert.")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[House Price MLOps] {subject}"
        msg["From"] = SMTP_EMAIL
        msg["To"] = ", ".join(recipients)

        content_type = "html" if html else "plain"
        msg.attach(MIMEText(body, content_type))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, recipients, msg.as_string())

        logger.info("Alert email sent to %s: %s", recipients, subject)
        return True

    except Exception as e:
        logger.error("Failed to send alert email: %s", e)
        return False


def send_training_success_alert(metrics: dict) -> bool:
    """Send email notification after successful training."""
    subject = "Training Completed Successfully"

    rmse = metrics.get("rmse_price")
    mae = metrics.get("mae_price")
    r2 = metrics.get("r2_price")
    mape = metrics.get("mape")

    body = f"""
    <h2>Model Training Completed</h2>
    <table border="1" cellpadding="8" cellspacing="0">
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>RMSE (price)</td><td>${rmse:,.2f}</td></tr>
        <tr><td>MAE (price)</td><td>${mae:,.2f}</td></tr>
        <tr><td>R2 (price)</td><td>{r2}</td></tr>
        <tr><td>MAPE</td><td>{mape}%</td></tr>
    </table>
    <p>Check MLflow for details: <a href="{MLFLOW_TRACKING_URI}">MLflow Dashboard</a></p>
    """
    return send_alert_email(subject, body, html=True)


def send_training_failure_alert(error: str) -> bool:
    """Send email notification when training fails."""
    subject = "Training FAILED"
    body = f"""
    <h2 style="color: red;">Model Training Failed</h2>
    <p><strong>Error:</strong></p>
    <pre>{error}</pre>
    <p>Please check the Airflow logs and fix the issue.</p>
    """
    return send_alert_email(subject, body, html=True)
