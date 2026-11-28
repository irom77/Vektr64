"""Running the trainer project from `vektr-ops`."""


def trainer_cmd(env: dict, module: str, *args: str) -> list[str]:
    """Command that runs a trainer module: the python named by `VEKTR_TRAINER_PYTHON`, else `uv run`."""
    python = env.get("VEKTR_TRAINER_PYTHON")
    if python:
        return [python, "-m", module, *args]
    return ["uv", "run", "--project", "trainer", "python", "-m", module, *args]
