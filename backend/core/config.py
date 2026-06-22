from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str

    HOST: str
    PORT: int

    UPLOAD_DIR: str
    REPORT_DIR: str
    MODEL_DIR: str

    DATABASE_URL: str

    REDIS_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()