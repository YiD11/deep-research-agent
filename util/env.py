import os


def get_environment():
    """
    Get the system environment from the OS environment variable DEEP_RESEARCH_AGENT_ENV.
    """
    env = os.getenv("DEEP_RESEARCH_AGENT_ENV")
    assert env is not None, "Environment variable DEEP_RESEARCH_AGENT_ENV is not set"
    return env


def is_dev():
    """
    Check if the system environment is DEV.
    """
    env = get_environment()
    return env == "DEV" or env == "DEBUG"


def is_debug():
    """
    Check if the system environment is DEBUG.
    """
    env = get_environment()
    return env == "DEBUG"
