"""Keep-invalid configurations for geometry reference captures."""


def keep_config(config_type, **kwargs):
    # The capture tools also run against the preceding release's config schema.
    if "on_invalid" in config_type.__dataclass_fields__:
        kwargs.setdefault("on_invalid", "keep")
    return config_type(**kwargs)
