# Sistema de Consulta de Viajes con RAG y Agente Inteligente

## Descripción General

Este proyecto implementa un sistema avanzado de consulta sobre catálogos de viajes utilizando las técnicas de Recuperación Aumentada de Generación (RAG) y un agente inteligente basado en LLM. El sistema permite realizar consultas en lenguaje natural sobre ofertas de viajes y obtener respuestas precisas basadas en la información contenida en catálogos de diversas agencias de viajes.

## Componentes Principales

### 1. Sistema RAG (Retrieval-Augmented Generation)

El componente RAG (`rag.py`) implementa las siguientes funcionalidades:

- **Procesamiento de Documentos**: Divide los catálogos de viajes en fragmentos (chunks) de tamaño configurable con solapamiento para mantener el contexto.
- **Generación de Embeddings**: Convierte cada fragmento de texto en vectores de embeddings utilizando el modelo `text-embedding-3-small` de OpenAI.
- **Almacenamiento Vectorial**: Guarda los embeddings en una base de datos SQLite con la extensión `sqlite-vec` para búsquedas por similitud.
- **Caché de Embeddings**: Implementa un sistema de caché para evitar generar embeddings repetidos y reducir costes.
- **Búsqueda Semántica**: Permite encontrar los fragmentos más relevantes para una consulta mediante similitud coseno.

### 2. Agente Inteligente

El agente (`agent.py`) actúa como intermediario entre el usuario y el sistema RAG:

- **Interpretación de Consultas**: Analiza las consultas del usuario y determina si están relacionadas con viajes.
- **Uso de Herramientas**: Utiliza la herramienta `VIAJES()` para acceder al sistema RAG cuando la consulta es pertinente.
- **Validación de Respuestas**: Verifica la calidad y relevancia de las respuestas mediante un sistema de validación.
- **Mejora Automática**: Si una respuesta no es satisfactoria, activa el modo de búsqueda mejorada.

## Búsqueda Mejorada

La funcionalidad de búsqueda mejorada es una característica destacada del sistema que permite obtener resultados más precisos en consultas complejas:

### Funcionamiento

1. **Generación de Respuestas Hipotéticas**: El sistema genera múltiples respuestas hipotéticas a la consulta original utilizando GPT-3.5 Turbo.
   - La función `generar_respuestas_hipoteticas()` solicita al modelo que genere varias posibles respuestas a la pregunta del usuario.
   - Estas respuestas se extraen mediante expresiones regulares de bloques de código Python.
   - El objetivo es ampliar el espectro semántico de la búsqueda original.

2. **Expansión de la Búsqueda**: Realiza búsquedas adicionales utilizando tanto la consulta original como las respuestas hipotéticas.
   - Por cada respuesta hipotética, se realiza una búsqueda adicional de fragmentos relevantes.
   - La función `buscar_chunks_similares()` se ejecuta múltiples veces con diferentes entradas.
   - Esto permite capturar información relevante que podría haberse perdido con una única consulta.

3. **Consolidación de Resultados**: Combina y elimina duplicados entre todos los fragmentos recuperados.
   - Se crea un diccionario `chunks_unicos` donde la clave es el contenido del chunk.
   - Si un chunk aparece más de una vez, se mantiene la versión con menor distancia (mayor similitud).
   - Este proceso garantiza que no haya información redundante en el contexto final.

4. **Reordenamiento por Relevancia**: Ordena los resultados consolidados por su distancia para seleccionar los mejores.
   - Los chunks se ordenan por su valor de distancia (menor distancia = mayor relevancia).
   - Se limita el número de chunks al valor especificado por el parámetro `max_chunks`.
   - Este paso asegura que solo se utilicen los fragmentos más relevantes para la generación de la respuesta.

5. **Generación de Respuesta Final**: Utiliza GPT-4 para generar una respuesta completa basada en los fragmentos más relevantes.
   - Se construye un prompt que incluye la consulta original y los chunks seleccionados.
   - La respuesta final se genera utilizando el modelo GPT-4 con baja temperatura para mayor precisión.

### Ventajas

- **Mayor Cobertura**: Captura información relevante que podría perderse con una búsqueda simple.
- **Robustez Frente a Ambigüedades**: Maneja mejor consultas poco claras o con múltiples interpretaciones.
- **Calidad Mejorada**: Proporciona respuestas más completas y precisas al considerar múltiples perspectivas.
- **Recuperación Semántica Enriquecida**: Al ampliar el contexto semántico con respuestas hipotéticas, logra recuperar información relevante aunque utilice términos diferentes a los de la consulta original.
- **Adaptabilidad**: Se ajusta mejor a diferentes estilos de consulta y variaciones lingüísticas de los usuarios.

## Guía de Uso

### Requisitos Previos

1. Configurar la API key de OpenAI en el archivo `.env`:
   ```
   OPENAI_API_KEY=tu_api_key
   ```

2. Instalar las dependencias:
   ```bash
   uv sync
   ```

### Uso del RAG

El módulo RAG puede utilizarse directamente mediante línea de comandos:

```bash
python -m rag "¿Cuáles son las ofertas de viajes a Italia?"
```

#### Opciones Disponibles

- `-c, --chunk-size`: Tamaño de cada fragmento de texto (por defecto: 2000)
- `-o, --overlap`: Número de caracteres solapados entre fragmentos (por defecto: 400)
- `-m, --max-distance`: Umbral máximo de distancia (por defecto: 0.90)
- `-k, --max-chunks`: Número máximo de chunks a seleccionar (por defecto: 5)
- `-r, --responses`: Número de respuestas hipotéticas a generar (por defecto: 3)
- `-m, --mejorada`: Activar búsqueda mejorada (flag)
- `-f, --force`: Rehacer la Base de Datos de Embeddings (flag)
- `-d, --debug`: Activar modo depuración (flag)

### Uso del Agente

El agente proporciona una interfaz más natural para interactuar con el sistema:

```bash
python -m agent --prompt "Necesito información sobre viajes a Roma en verano"
```

#### Opciones Disponibles

- `--prompt, -p`: La consulta del usuario (requerido)
- `-d, --debug`: Activar modo depuración (flag)

### Ejemplos de Consultas

```bash
# Búsqueda simple con RAG
python -m rag "¿Qué tours hay disponibles para Italia?"

# Búsqueda mejorada con RAG
python -m rag "¿Cuáles son las mejores opciones para viajar a ciudades históricas en Europa?" -m

# Consulta mediante el agente
python -m agent --prompt "Busco un viaje para familia con niños en destinos de playa"
```

## Estructura de Datos

El sistema utiliza dos bases de datos SQLite:

1. **embeddings.db**: Almacena los embeddings de los fragmentos de texto para búsqueda por similitud.
   - Estructura: Tabla virtual `embeddings` con campos `chunk` (texto) y `embedding` (vector float[1536])
   - Permite búsquedas por similitud mediante la extensión sqlite-vec

2. **embedding_cache.db**: Caché de embeddings para evitar regenerar vectores para textos ya procesados.
   - Estructura: Tabla `embedding_cache` con campos:
     - `text_hash`: Hash MD5 del texto (PRIMARY KEY)
     - `text`: Texto completo
     - `embedding`: Embedding serializado
     - `created_at`: Fecha de creación

## Flujo de Funcionamiento del Sistema

### Proceso RAG Básico

1. **Ingestión de datos**:
   - Lectura de archivos markdown desde el directorio especificado.
   - División en chunks con solapamiento configurable.
   - Generación y almacenamiento de embeddings.

2. **Procesamiento de consultas**:
   - Generación de embedding para la consulta del usuario.
   - Búsqueda de chunks similares mediante similitud coseno.
   - Construcción de prompt con chunks relevantes.
   - Generación de respuesta utilizando OpenAI.

### Proceso con Agente

1. **Recepción de consulta**:
   - El usuario envía una consulta al agente.
   - El agente evalúa si la consulta está relacionada con viajes.

2. **Delegación a RAG**:
   - Si es relevante, el agente llama a la herramienta `VIAJES()`.
   - Esta herramienta utiliza el sistema RAG para obtener una respuesta.

3. **Validación**:
   - El agente verifica si la respuesta es satisfactoria.
   - Si no lo es, activa automáticamente la búsqueda mejorada.

4. **Presentación**:
   - El agente formatea y presenta la respuesta final al usuario.

## Posibles Mejoras

- **Interfaz Web**: Implementación de una interfaz web para facilitar la interacción con el sistema.
- **Expansión de Fuentes**: Soporte para más fuentes de datos y formatos de documentos.
- **Refinamiento del Chunking**: Mejoras en el proceso de chunking para optimizar la calidad de las respuestas.
- **Retroalimentación del Usuario**: Sistema que permita al usuario calificar las respuestas para mejorar resultados futuros.
- **Optimización de Parámetros**: Ajuste automático de parámetros como tamaño de chunk y umbral de distancia.
- **Filtros Temáticos**: Implementación de filtros por categorías de viajes (playa, montaña, ciudad, etc.).
- **Multimodalidad**: Integración con análisis de imágenes para procesar catálogos con contenido visual. 