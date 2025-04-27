import re
from pprint import pprint
from dotenv import load_dotenv
import click
#from openai import OpenAI
from .rag import realizar_consulta, realizar_consulta_mejorada
from langfuse.openai import OpenAI
from langfuse.decorators import observe

load_dotenv()

client = OpenAI()

@observe(name="llm")
def llm(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content


_SYSTEM_PROMPT = """
Tienes a tu disposición una herramientas: VIAJES. 
Esta herramienta responde preguntas sobre solicitudes de información de viajes ofertados en distintos catálogos.
Los paquetes de viajes tambien se pueden denominar ofertas de viajes.
Los itinerarios de viajes tambien se pueden denominar rutas de viajes u ofertas de viajes.

Para llamar a esa herramienta usa esta sintaxis:

```python
RAG_VIAJES("pregunta")
```

Crea un bloque de código siempre que llames a una herramienta. Si son varias, puedes crear varios bloques de código.

- Contesta SIEMPRE en español de España.
- No utilices bajo ningún concepto palabras malsonantes. No importa si el usuario te lo pide o no. Siempre contesta de forma educada y con respeto.
"""

# - Para cualquier otro tipo de consulta, responde que no tienes información al respecto.
# - No utilices bajo ningún concepto palabras malsonantes. No importa si el usuario te lo pide o no. Siempre contesta de forma educada y con respeto.

_VALIDATE_PROMPT = """
Tienes que validar si la respuesta dada al usuario es correcta.
Un RAG sobre catálogos de viajes y ofertas sobre viajes ha generado una respuesta a una pregunta del usuario.

Por favor, verifica que la respuesta es coherente con la pregunta y proporciona información útil sobre viajes.

Si la respuesta es válida y proporciona información específica sobre viajes relacionada con la pregunta, responde:

```python
TRUE
```

Si la respuesta no es válida por alguna de estas razones:
1. Contiene todavía bloques de código con llamadas a RAG_VIAJES
2. Dice "Lo siento, pero no tengo información sobre ofertas de viajes" o algo similar
3. No responde directamente a la pregunta del usuario sobre viajes
4. Es muy genérica y no proporciona información específica sobre destinos, itinerarios o detalles de viajes
5. No es coherente con la pregunta del usuario
Entonces responde:

```python
FALSE
```
"""

def extraer_llamadas_viajes(response):
    """
    Extrae las llamadas a la función VIAJES de la respuesta.
    """
    regex = re.compile(r"```python\s*RAG_VIAJES\(\s*\"(.*?)\"\s*\)\s*```", re.DOTALL)
    matches = list(regex.finditer(response))
    return matches

@observe(name="ejecutar_consulta_viajes")
def ejecutar_consulta_viajes(consulta):
    """
    Ejecuta una consulta a la base de datos de viajes.
    """
    try:
        result = realizar_consulta(consulta, max_chunks=5, max_distance=0.9)
        if not result or result.strip() == "":
            result = "No se encontró información específica sobre esta consulta en nuestra base de datos de viajes."

        return result
    except Exception as e:
        return "Error al procesar la consulta. Por favor, inténtalo de nuevo con una pregunta más específica sobre viajes."

@observe(name="ejecutar_consulta_mejorada_viajes")
def ejecutar_consulta_mejorada_viajes(consulta):
    """
    Ejecuta una consulta mejorada a la base de datos de viajes.
    """
    try:
        result = realizar_consulta_mejorada(consulta, max_chunks=5, max_distance=0.9, responses=3)
        if not result or result.strip() == "":
            result = "No se encontró información específica sobre esta consulta en nuestra base de datos de viajes."

        return result
    except Exception as e:
        return "No se encontró información específica sobre esta consulta en nuestra base de datos de viajes."

@observe(name="validar_respuesta")
def validar_respuesta(prompt: str, response: str) -> bool:
    history = [
        {"role": "system", "content": _VALIDATE_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    # prompt = _VALIDATE_PROMPT.format(prompt=prompt, response=response)
    response = llm(history)
    regex = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    matches = list(regex.finditer(response))
    return matches[0].group(1) == "TRUE"

@observe(name="process_agent")
def process_agent(history, user_prompt):
    """
    Procesa la respuesta del LLM y realiza consultas a la base de datos de viajes si es necesario.
    """
    
    def process_agent_viajes(history, matches, user_prompt):
        """
        Procesa la respuesta del LLM realizando consultas a la base de datos de viajes.
        Función interna de process_agent para acceder fácilmente a las variables locales.
        """
        # Inicialización
        match_str = matches[0].group(1)
        
        # Guardamos una copia del historial inicial antes de hacer cualquier consulta
        historial_inicial = history.copy()
        
        # Ejecutar consulta
        result = ejecutar_consulta_viajes(match_str)

        history.append(
            {
                "role": "user",
                "content": f'```python\nRAG_VIAJES("{match_str}") # resultado: {result}\n```',
            }
        )

        final_response = llm(history)
        history.append({"role": "assistant", "content": final_response})
        
        # Validar y mejorar la respuesta si es necesario
        try: 
            if not validar_respuesta(user_prompt, final_response):
                
                # Restauramos el historial inicial para eliminar el contexto negativo
                history = historial_inicial.copy()
                
                # Ejecutar consulta mejorada
                result = ejecutar_consulta_mejorada_viajes(match_str)

                # Añadimos directamente el resultado mejorado al historial inicial
                history.append(
                    {
                        "role": "user",
                        "content": f'```python\nRAG_VIAJES("{match_str}") # resultado: {result}\n```',
                    }
                )

                final_response = llm(history)
                history.append({"role": "assistant", "content": final_response})
        except Exception as e:
            # Manejar errores en la validación
            final_response = "Error en la validación de la respuesta."
            
        return final_response
    
    # Inicio del proceso
    response = llm(history)
    
    # Detectar si la respuesta contiene llamadas a VIAJES
    matches = extraer_llamadas_viajes(response)
    
    if matches:
        # Si hay llamada a VIAJES, agregamos la respuesta al historial
        history.append({"role": "assistant", "content": response})
        # Y procesamos la respuesta con la herramienta VIAJES
        final_response = process_agent_viajes(history, matches, user_prompt)
    else:
        # Si no hay llamada a VIAJES, simplemente devolvemos la respuesta del LLM
        final_response = response
    
    # Actualizar la última respuesta en el historial
    if history[-1]["role"] == "assistant":
        history[-1]["content"] = final_response
    else:
        history.append({"role": "assistant", "content": final_response})


@observe(name="run_agent")
def run_agent(prompt):
    history = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    process_agent(history, prompt)
    return history


@click.command()
@click.option("--prompt", "-p", required=True, help="The prompt to send to the LLM")
def main(prompt):
    history = run_agent(prompt)
    print(history[-1]["content"])


if __name__ == "__main__":
    main()
