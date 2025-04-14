# Ejercicio de RAG básico

## Objetivo

Escribir de 0 sin ayudas de frameworks o bibliotecas de RAG. Un RAG que:

- Haga chunking de un documento. Aquí como ejemplo es usará un catalogo de viajes..
  Hemos usado `markitdown` para extraer el texto de los PDFs.

  - Si quieres usar otro(s) documentos te recomiendo que busques un ejemplo
    "sucio": que tenga errores, párrafos mal formateados, etc. En el caso del
    BOE tenemos buena calidad de texto pero algunas cosas son incorrectas: por
    ejemplo en los bordes de las páginas aparece texto en vertical que en
    markdown genera "ruido".

- Haga embedding de los chunks. Usa la API de
  [OpenAI para hacer embedding](https://platform.openai.com/docs/guides/embeddings)
  de los chunks.

- Suba todos los chunks a una base de datos (ver sección "Base de datos para
  RAG")

- Haga consultas a la BBDD con la query del usario

- Cree un prompt con el contexto obtenido, la pregunta del usuario y una
  instrucción para que el LLM responda.

- Mostrar la respuesta al usuario: os aconsejo que la pregunta se pueda
  proporcionar como argumento de la línea de comandos para facilitar las
  pruebas.

### Extra

Si queréis profundizar más...

- Permitir modificar el tamaño de los chunks y su overlap mediante argumentos de
  la línea de comandos.
  - Si hacéis esto, os aconsejo que cacheéis el embedding (saltar si ya está en
    la BBDD) para ahorrar costes.
- Haz pruebas con distintos tamaños de chunk para ver cuál es mejor para vuestro
  caso de uso.
- Permitir ingestar más documentos

## Base de datos para RAG

- SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) ⭐️⭐️
- [Supabase](https://supabase.com/) (PostgreSQL administrado en la nube) ⭐️
- [Pinecone](https://www.pinecone.io/) (vector database en la nube) ⭐️
- PostgreSQL + [pgvector](https://github.com/pgvector/pgvector)
- Usar python directamente y calcular las distancias con `scipy` o `faiss`
  (recomendado sólo si queréis aprender cómo funciona por dentro)

⭐️ = recomendado

## Solución

Se colgará la semana que viene :)

# Uso

## Instalación del entorno

```bash
uv sync
```

## Ejecución

```bash
python -m rag
```

# Dudas

Para cualquier duda por favor, escribid en el foro del campus.
