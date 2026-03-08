from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    random_state: int = 42
    target_column: str = "y"
    id_column: str = "id"


settings = Settings()