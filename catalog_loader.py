import yaml
 
def load_model_catalog(path="model_catalog.yaml"):
    """Load the model catalog YAML file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model catalog file not found at {path}")
 
def get_s3_config(catalog):
    """Fetch S3 bucket and folder config from the catalog."""
    s3 = catalog.get("s3_config", {})
    if not s3:
        raise ValueError("'s3_config' section not found in model_catalog.yaml")
    return s3.get("bucket_name"), s3.get("folders", {})

# def get_production_model_info(catalog, project_name):
#     """Fetch model details for a given project from the catalog."""
#     project_info = catalog.get("production_models", {}).get(project_name)
#     if not project_info:
#         raise ValueError(f"Project '{project_name}' not found in production catalog.")
#     return {
#         "model": project_info.get("model"),
#         "task_type": project_info.get("task_type"),
#         "hyperparameters": project_info.get("hyperparameters", {}),
#         "metrics": project_info.get("metrics", [])
#     }
