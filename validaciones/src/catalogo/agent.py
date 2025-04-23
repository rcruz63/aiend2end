import re
from pprint import pprint
from dotenv import load_dotenv
import click
from openai import OpenAI
from .rag import realizar_consulta, realizar_consulta_mejorada

load_dotenv()

client = OpenAI()

def dprint(mess: str, debug: bool = False):
    """
    Imprime los mensajes solo si estamos en modo debug
    """
    if debug:
        pprint(mess)


def llm(messages: list[dict], debug: bool = False) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    dprint(f"Respuesta LLM: {response.choices[0].message.content}", debug)
    dprint(response, debug)
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


def ejecutar_consulta_viajes(consulta, debug=False):
    """
    Ejecuta una consulta a la base de datos de viajes.
    """
    try:
        result = realizar_consulta(consulta, max_chunks=5, max_distance=0.9, debug=debug)
        if not result or result.strip() == "":
            result = "No se encontró información específica sobre esta consulta en nuestra base de datos de viajes."

        return result
    except Exception as e:
        dprint(f"Error al ejecutar la consulta: {str(e)}", debug)
        return "Error al procesar la consulta. Por favor, inténtalo de nuevo con una pregunta más específica sobre viajes."


def ejecutar_consulta_mejorada_viajes(consulta, debug=False):
    """
    Ejecuta una consulta mejorada a la base de datos de viajes.
    """
    try:
        result = realizar_consulta_mejorada(consulta, max_chunks=5, max_distance=0.9, responses=3, debug=debug)
        if not result or result.strip() == "":
            result = "No se encontró información específica sobre esta consulta en nuestra base de datos de viajes."

        return result
    except Exception as e:
        dprint(f"Error al ejecutar la consulta mejorada: {str(e)}", debug)
        return "No se encontró información específica sobre esta consulta en nuestra base de datos de viajes."


def validar_respuesta(prompt: str, response: str, debug: bool = False) -> bool:
    history = [
        {"role": "system", "content": _VALIDATE_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    # prompt = _VALIDATE_PROMPT.format(prompt=prompt, response=response)
    response = llm(history, debug)
    regex = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    matches = list(regex.finditer(response))
    return matches[0].group(1) == "TRUE"


def process_agent(history, user_prompt, debug: bool = False):
    """
    Procesa la respuesta del LLM y realiza consultas a la base de datos de viajes si es necesario.
    """
    
    def process_agent_viajes(history, matches, user_prompt, debug=False):
        """
        Procesa la respuesta del LLM realizando consultas a la base de datos de viajes.
        Función interna de process_agent para acceder fácilmente a las variables locales.
        """
        # Inicialización
        match_str = matches[0].group(1)
        dprint(f"Consulta extraída: {match_str}", debug)
        
        # Guardamos una copia del historial inicial antes de hacer cualquier consulta
        historial_inicial = history.copy()
        
        # Ejecutar consulta
        result = ejecutar_consulta_viajes(match_str, debug)

        dprint(f"Resultado de la consulta normal: {result}", debug)

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
            if not validar_respuesta(user_prompt, final_response, debug):
                dprint(f"** MEJORANDO RESPUESTA **: {final_response}", debug)
                
                # Restauramos el historial inicial para eliminar el contexto negativo
                history = historial_inicial.copy()
                
                # Ejecutar consulta mejorada
                result = ejecutar_consulta_mejorada_viajes(match_str, debug)
                
                dprint(f"Resultado de la consulta mejorada: {result}", debug)

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
            dprint(f"Error en la validación: {str(e)}", debug)
            
        return final_response
    
    # Inicio del proceso
    dprint(f"Historial pre-llm: {history}", debug)
    response = llm(history, debug)
    dprint(f"Respuesta del LLM Inicial: {response}", debug)
    
    # Detectar si la respuesta contiene llamadas a VIAJES
    matches = extraer_llamadas_viajes(response)
    
    if matches:
        # Si hay llamada a VIAJES, agregamos la respuesta al historial
        history.append({"role": "assistant", "content": response})
        # Y procesamos la respuesta con la herramienta VIAJES
        final_response = process_agent_viajes(history, matches, user_prompt, debug)
    else:
        # Si no hay llamada a VIAJES, simplemente devolvemos la respuesta del LLM
        final_response = response
    
    # Actualizar la última respuesta en el historial
    if history[-1]["role"] == "assistant":
        history[-1]["content"] = final_response
    else:
        history.append({"role": "assistant", "content": final_response})
    
    dprint(f"Historial post-procesamiento: {history}", debug)


def run_agent(prompt, debug: bool = False):
    history = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    process_agent(history, prompt, debug)
    return history


@click.command()
@click.option("--prompt", "-p", required=True, help="The prompt to send to the LLM")
@click.option('-d', '--debug', is_flag=True, default=False, help='Activar modo depuración')
def main(prompt, debug):
    history = run_agent(prompt, debug)
    print(history[-1]["content"])


if __name__ == "__main__":
    main()
