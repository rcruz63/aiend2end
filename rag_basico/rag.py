import os
import openai
from dotenv import load_dotenv
import glob
from pathlib import Path
from tqdm import tqdm
import sqlite3
import sqlite_vec
import json
import hashlib
import click

load_dotenv()

# Función para imprimir mensajes de depuración
def dprint(text: str, debug: bool = False):
    """
    Imprime mensajes de depuración solo si el modo debug está activado.
    
    Args:
        text (str): El texto a imprimir
        debug (bool): Si True, imprime el mensaje; si False, no hace nada
    """
    if debug:
        print(text)

titulos = {
        "cam_25_halcon_viajes_39_mb_1.md": "Halcon Viajes",
        "cam_nacional_e_internacional_2025.md": "NAUTALIA Viajes",
        "CATALOGO_RUTAS_CAM_2025_compressed.md": "B Travel | Carrefour Viajes",
        "rc25_cibeles_v2.md": "Cibeles",
        "rc25_veci_ok.md": "Viajes El Corte Inglés",
        "rutas_25_bttb_v4.md": "B Travel",
        "rutas25_iag7_v3.md": "IAG7 Viajes",
    }


def chunker(text: str, filename: str = "", chunk_size: int = 2000, overlap: int = 400) -> list[str]:
    """
    Divide un texto en fragmentos de un tamaño especificado, con un solapamiento entre fragmentos.
    Añade el nombre del archivo como primera línea de cada fragmento.

    Args:
        text (str): El texto a dividir.
        filename (str): Nombre del archivo de origen para incluir en cada fragmento.
        chunk_size (int): El tamaño de cada fragmento.
        overlap (int): El número de caracteres solapados entre fragmentos.

    Returns:
        list[str]: Una lista de fragmentos del texto con el nombre del archivo como primera línea.
    """

    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i : i + chunk_size]
        # Añadir el nombre del archivo como primera línea
        if filename:
            chunk_with_metadata = f"Esto es un fragmento del catálogo de viajes de la agencia: {titulos[filename]}\n{chunk_text}"
        else:
            chunk_with_metadata = chunk_text
        chunks.append(chunk_with_metadata)
    return chunks

def get_text_hash(text: str) -> str:
    """
    Genera un hash único para un texto.
    
    Args:
        text (str): El texto a hashear.
        
    Returns:
        str: El hash del texto.
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def init_cache_db():
    """
    Inicializa la base de datos de caché para embeddings.
    
    Returns:
        sqlite3.Connection: Conexión a la base de datos de caché.
    """
    conn = sqlite3.connect("embedding_cache.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embedding_cache (
            text_hash TEXT PRIMARY KEY, 
            text TEXT, 
            embedding BLOB, 
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)
    ''')
    
    conn.commit()
    return conn

def load_sqlite_vec(conn):
    """
    Carga la extensión sqlite-vec para permitir cálculos de similitud de vectores.
    
    Args:
        conn (sqlite3.Connection): Conexión a la base de datos.
    """
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

def get_embeddings(text: str, debug: bool = False):
    """
    Genera un embedding para un texto utilizando la API de OpenAI.
    Si el texto ya existe en la caché, devuelve el embedding almacenado.
    
    Args:
        text (str): El texto para el cual generar el embedding.
        debug (bool): Si True, muestra mensajes de depuración.
        
    Returns:
        list[float]: El vector embedding generado por el modelo.
    """
    try:
        text_hash = get_text_hash(text)
        
        # Verificar si existe en la caché
        conn = init_cache_db()
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM embedding_cache WHERE text_hash = ?", (text_hash,))
        cached_result = cursor.fetchone()
        
        if cached_result:
            dprint(f"Usando embedding almacenado en caché para el texto: {text[:50]}...", debug)
            conn.close()
            return json.loads(cached_result[0])
        
        # Si no está en caché, generar nuevo embedding
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No se encontró la API key de OpenAI. Configúrela en el archivo .env")
        
        client = openai.OpenAI(api_key=api_key)
        
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Modelo de embeddings más reciente
            input=text
        )
        
        embedding = response.data[0].embedding

        dprint(f"Guardando embedding en caché para el texto: {text[:50]}...", debug)
        # Guardar en caché
        
        cursor.execute("INSERT INTO embedding_cache (text_hash, text, embedding) VALUES (?, ?, ?)", 
                      (text_hash, text, json.dumps(embedding)))
        conn.commit()
        conn.close()
        
        return embedding
    
    except Exception as e:
        print(f"Error al generar embedding: {e}")
        if 'conn' in locals() and conn:
            conn.close()
        return None

def populate_embeddings(chunks: list[str], debug: bool = False):
    """
    Poblar la base de datos con los embeddings de los chunks.
    
    Args:
        chunks (list[str]): Lista de fragmentos de texto para generar embeddings.
        debug (bool): Si True, muestra mensajes de depuración.
    """

    # Conectar a la base de datos (se creará si no existe)
    conn = sqlite3.connect("embeddings.db")

    # Cargar la extensión sqlite-vec
    load_sqlite_vec(conn)
    
    cursor = conn.cursor()
    
    # Crear tabla si no existe - usando BLOB para almacenar los embeddings
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS embeddings 
        using vec0(chunk TEXT, embedding float[1536])
    ''')
    conn.commit()

    conn.execute("BEGIN TRANSACTION")

    for chunk in tqdm(chunks, desc="Generando embeddings"):
        embedding = get_embeddings(chunk, debug=debug)
        cursor.execute(
            "INSERT INTO embeddings (embedding, chunk) VALUES (?, ?)",
                (sqlite_vec.serialize_float32(embedding), chunk),
            )
        
    # Commit y cierre de conexión
    conn.commit()
    conn.close()
    dprint(f"Se almacenaron {len(chunks)} embeddings en la base de datos", debug)
    return True
    

    
def get_embeddings_query(query: str):
    """
    Genera un embedding para una query utilizando la API de OpenAI.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró la API key de OpenAI. Configúrela en el archivo .env")
    
    client = openai.OpenAI(api_key=api_key)
    
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Modelo de embeddings más reciente
        input=query
    )

    return response.data[0].embedding

def buscar_chunks_similares(query: str, max_chunks: int = 5, max_distance=0.95, debug: bool = False):
    """
    Busca los chunks más similares a un embedding utilizando la función cosine_similarity.
    
    Args:
        query(str): La consulta.
        max_chunks (int): Número máximo de chunks a devolver.
        max_distance (float): Umbral máximo de distancia (0-1).
        debug (bool): Si True, muestra mensajes de depuración.
        
    Returns:
        list: Los chunks más similares con sus puntuaciones de similitud.
    """

    query_embedding = get_embeddings_query(query)

    conn = sqlite3.connect("embeddings.db")
    
    # Cargar la extensión sqlite-vec
    load_sqlite_vec(conn)
    
    cursor = conn.cursor()
    
    # Ejecutar la consulta SQL para obtener los chunks similares usando cosine_similarity
    dprint("Ejecutando consulta SQL para buscar chunks similares...", debug)
    
    # Primero, vamos a comprobar cuántos chunks hay en total en la base de datos
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    total_chunks = cursor.fetchone()[0]
    dprint(f"Total de chunks en la base de datos: {total_chunks}", debug)
    
    # Ahora ejecutamos la consulta de similitud pero sin filtros
    cursor.execute(
        """
            select
                chunk,
                distance
            from embeddings
            where embedding match ?
            and distance < ?
            and k = ?
            order by distance;
        """,
        (sqlite_vec.serialize_float32(query_embedding), max_distance, max_chunks),
    )
    
    # Obtener los resultados    
    results = cursor.fetchall()
    dprint(f"Chunks encontrados antes de filtrar: {len(results)}", debug)
    
    # Mostrar las distancias para diagnóstico
    dprint("Distancias de los chunks encontrados:", debug)
    for i, (_, distance) in enumerate(results):
        similarity = 1 - distance
        dprint(f"Chunk {i+1}: Distancia={distance:.4f}, Similitud={similarity:.4f}", debug)
    
    conn.close()
    
    # Filtrar por similitud si es necesario (1-distance para convertir distancia a similitud)
    filtered_results = [(chunk, distance) for chunk, distance in results if distance < max_distance]
    dprint(f"Chunks que pasan el filtro de distancia ({max_distance}): {len(filtered_results)}", debug)
    
    return filtered_results

def crear_prompt(query: str, resultados: list[tuple[str, float]], max_chunks: int = 5) -> str:
    """
    Crea el prompt para OpenAI combinando la consulta y el contexto relevante.
    
    Args:
        query (str): Pregunta o consulta del usuario
        resultados (list[Dict[str, Any]]): Lista de chunks relevantes encontrados,
            cada uno debe contener ruta_archivo y contenido
        
    Returns:
        str: Prompt formateado para enviar a OpenAI
    """
    # Formatear el contexto de los chunks
    contexto = ("-"*100).join([
        f"\n{chunk}\n"
        for chunk, _ in resultados[:max_chunks]
    ])
    
    # Construir el prompt
    return f"""El usuario necesita que respondas la siguiente pregunta basandote solo en el contexto proporcionado.
Si la información en el contexto no es suficiente para responder, indícalo claramente.

Usuario: {query}

Contexto relevante:
{contexto}

Respuesta:"""

def obtener_respuesta_openai(prompt: str, test_mode: bool = False) -> str:
    """
    Obtiene una respuesta de OpenAI usando el modelo GPT-4.
    
    Args:
        prompt (str): Prompt completo con pregunta y contexto
        test_mode (bool): Si True, muestra información de depuración y tiempos
        
    Returns:
        str: Respuesta generada por OpenAI
        
    Raises:
        Exception: Si hay un error en la llamada a la API de OpenAI
    """
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un asistente experto que responde preguntas basándose únicamente en el contexto proporcionado."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    
    respuesta = response.choices[0].message.content
    
    return respuesta


def realizar_consulta(query: str, max_chunks: int = 5, max_distance: float = 0.95, debug: bool = False):
    """
    Realiza una consulta al sistema de RAG.

    Primero se genera un embedding de la query.
    Luego se busca el embedding en la base de datos de embeddings utilizando la función cosine_similarity.
    Se obtienen los chunks más similares.
    Se contruye un prompt con los chunks más similares.
    Se realiza una consulta a OpenAI con el prompt.
    Se devuelve el resultado de la consulta.
    
    Args:
        query (str): La consulta del usuario.
        max_chunks (int): Número máximo de chunks a seleccionar.
        max_distance (float): Umbral máximo de distancia.
        debug (bool): Si True, muestra mensajes de depuración.
        
    Returns:
        str: La respuesta generada por OpenAI.
    """
    
    # Buscar chunks similares pasando directamente la query como texto
    similar_chunks = buscar_chunks_similares(query, max_chunks=max_chunks, max_distance=max_distance, debug=debug)

    dprint(f"Se encontraron {len(similar_chunks)} chunks similares", debug)
    # Mostramos las primeras 5 líneas de cada chunk y su similitud
    for chunk, distance in similar_chunks[:5]:
        dprint(f"Distancia: {distance:.4f}", debug)
        dprint(chunk[:500], debug)
        dprint("-"*100, debug)

    prompt = crear_prompt(query, similar_chunks, max_chunks=max_chunks)

    respuesta = obtener_respuesta_openai(prompt)

    return respuesta


def get_all_file_paths(directory: str=".") -> list[str]:
    """

    Retorna una lista de rutas a todos los archivos `.md` en el directorio especificado.

    Args:
        directory (str): El directorio a buscar archivos `.md`. Por defecto es el directorio actual.

    Returns:
        list[str]: Una lista de rutas a todos los archivos `.md` en el directorio especificado.
    """

    file_pattern = os.path.join(directory, "*.md")
    md_files = glob.glob(file_pattern)
    return [Path(f) for f in md_files]

def read_files(directory: str=".", debug: bool = False) -> dict:
    """
    Lee el contenido de todos los archivos `.md` en el directorio especificado.
    
    Args:
        directory (str): El directorio a buscar archivos `.md`. Por defecto es el directorio actual.
        debug (bool): Si True, muestra mensajes de depuración.
        
    Returns:
        dict: Un diccionario con el nombre como clave y el contenido como valor. 
    """

    file_paths = get_all_file_paths(directory)

    md_files = {}

    for file_path in file_paths:
        md_files[file_path.name] = read_file(file_path)

    dprint(f"Se encontraron {len(md_files)} archivos .md: {list(md_files.keys())}", debug)
    return md_files

def read_file(file_path: Path) -> str:
    """
    Lee el contenido de un archivo `.md` y lo retorna como una cadena de texto.

    Args:
        file_path (str): La ruta del archivo `.md` a leer.

    Returns:
        str: El contenido del archivo `.md` como una cadena de texto.
    """

    with open(file_path, "r") as file:
        return file.read()
    
def chunk_files(files: dict, chunk_size: int = 2000, overlap: int = 400) -> list[str]:
    """
    Divide el contenido de cada archivo en fragmentos caracteres, con solapamiento.

    Args:
        files (dict): Un diccionario con el nombre como clave y el contenido como valor.
        chunk_size (int): El tamaño de cada fragmento.
        overlap (int): El número de caracteres solapados entre fragmentos.

    Returns:
        list[str]: Una lista de fragmentos de texto, cada uno con el nombre del archivo en la primera línea.
    """

    chunks = []
    for file_name, file_content in files.items():
        chunks.extend(chunker(file_content, file_name, chunk_size, overlap))

    return chunks

@click.command()
@click.argument('query', required=True)
@click.option('-c', '--chunk-size', default=2000, help='Tamaño de cada fragmento de texto')
@click.option('-o', '--overlap', default=400, help='Número de caracteres solapados entre fragmentos')
@click.option('-m', '--max-distance', default=0.95, help='Umbral maximo de distancia')
@click.option('-k', '--max-chunks', default=5, help='Número máximo de chunks a seleccionar')
@click.option('-f', '--force', is_flag=True, default=False, help='Rehacer la Base de Datos de Embeddings')
@click.option('-d', '--debug', is_flag=True, default=False, help='Activar modo depuración')
def main(query, chunk_size, overlap, max_distance, max_chunks, force, debug):
    """Inicia RAG básico con metadatos simples."""

    # Si no existe la base de datos de embeddings o se indica con force, se crea una nueva.
    if not os.path.exists("embeddings.db") or force:
        dprint("Rehaciendo la Base de Datos de Embeddings", debug)

        dprint("Iniciando RAG básico con metadatos simples", debug)
        dprint(f"Configuración: chunk_size={chunk_size}, overlap={overlap}", debug)
    
        # Leer archivos y obtener rutas
        files = read_files("./catalogo_md", debug=debug)
        
        # Dividir en chunks con metadatos
        chunks = chunk_files(files, chunk_size=chunk_size, overlap=overlap)

        populate_embeddings(chunks, debug=debug)
    
    # Realizar la consulta con la query proporcionada
    dprint(f"Realizando consulta: '{query}'", debug)
    respuesta = realizar_consulta(query, max_chunks=max_chunks, max_distance=max_distance, debug=debug)
    
    # Esta es la única impresión que no usamos dprint porque es la respuesta final al usuario
    print(respuesta)

if __name__ == "__main__":
    main()
