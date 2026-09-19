"""API routing package. Legacy optional modules are imported only on request."""

def register_blueprints(app):
    """Register the supplied v3 services on the legacy entry point."""
    from api.routes.v3 import create_v3_blueprint
    services = getattr(app, "extensions", {}).get("cevahir_services")
    if not services:
        raise RuntimeError("Use api.app_factory.create_app to initialize API services")
    app.register_blueprint(create_v3_blueprint(**services))
