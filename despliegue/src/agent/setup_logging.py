import logging
import os
from typing import Optional


def setup_logging(log_level: Optional[str] = None) -> None:

    # Obtener el nivel de logging de la variable de entorno o usar el proporcionado
    level = log_level or os.getenv('LOG_LEVEL', 'INFO')
    level_ext = log_level or os.getenv('LOG_LEVEL_EXT', logging.WARNING)

    # Configurar el formato del logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configurar el logging básico
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configurar el nivel de logging para el logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Configurar el nivel de logging para el logger de OpenAI
    openai_logger = logging.getLogger('openai')
    openai_logger.setLevel(level_ext)
    # Configurar el nivel de logging para el logger de Supabase
    supabase_logger = logging.getLogger('supabase')
    supabase_logger.setLevel(level_ext)
    # Configurar el nivel de logging para el logger de FastAPI
    fastapi_logger = logging.getLogger('fastapi')
    fastapi_logger.setLevel(level_ext)
    # Configurar el nivel de logging para el logger de uvicorn
    uvicorn_logger = logging.getLogger('uvicorn')
    uvicorn_logger.setLevel(level_ext)
