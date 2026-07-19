"""Application identity shared by API, clients, and release checks."""

APP_NAME = "Raven Watch Agents Pro"
APP_VERSION = "1.10.1"
PRODUCT_ID = "RavenWatchAgentsPro"
PRODUCT_USER_AGENT = f"{PRODUCT_ID}/{APP_VERSION}"


__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "PRODUCT_ID",
    "PRODUCT_USER_AGENT",
]
