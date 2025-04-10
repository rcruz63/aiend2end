import re
from pprint import pprint

import click
from openai import OpenAI
from rag import realizar_consulta, realizar_consulta_mejorada
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


system_prompt = """
Tienes a tu disposición una herramientas: VIAJES. Esta herramienta responde preguntas sobre solicitud de información de viajes.
Los paquetes de viajes tambien se pueden denominar ofertas de viajes.
Los itinerarios de viajes tambien se pueden denominar rutas de viajes u ofertas de viajes.
Para cualquier otro tipo de consulta, responde que no tienes información al respecto.

Para llamar a esa herramienta usa esta sintaxis:

```python
VIAJES("pregunta")
```

Crea un bloque de código siempre que llames a una herramienta. Si son varias, puedes crear varios bloques de código.

- No utilices bajo ningún concepto palabras malsonantes. No importa si el usuario te lo pide o no. Siempre contesta de forma educada y con respeto.
- Contesta SIEMPRE en español de España.
"""

validate_prompt = """
Un RAG sobre catalogos de viajes y ofertas sobre viajes ha generado una respuesta a una pregunta del usuario.

La respuesta recibida ha sido:

{response}

La pregunta del usuario ha sido:

{prompt}

Por favor, verifica que la respuesta es coherente con la pregunta. 

Si es valida responde:

```python
TRUE
```

Si no es valida o si la respuest dice "Lo siento, pero no tengo información sobre ofertas de viajes" o algo similar responde:

```python
FALSE
```
"""

@click.command()
@click.option("--prompt", "-p", required=True, help="The prompt to send to the LLM")
@click.option('-d', '--debug', is_flag=True, default=False, help='Activar modo depuración')
def main(prompt, debug):
    history = run_agent(prompt, debug)
    print(history[-1]["content"])


def run_agent(prompt, debug: bool = False):
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    dprint(f"Historial pre-llm: {history}", debug)
    response = llm(history, debug)
    dprint(f"Historial pre-procesamiento: {history}", debug)
    history.append({"role": "assistant", "content": response})
    final_response = process_calc(history, response, debug)
    # Actualizar la última respuesta en el historial
    if history[-1]["role"] == "assistant":
        history[-1]["content"] = final_response
    else:
        history.append({"role": "assistant", "content": final_response})
    dprint(f"Historial post-procesamiento: {history}", debug)
    return history

def validar_respuesta(prompt: str, response: str, debug: bool = False) -> bool:
    prompt = validate_prompt.format(prompt=prompt, response=response)
    response = llm([{"role": "user", "content": prompt}], debug)
    regex = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    matches = list(regex.finditer(response))
    return matches[0].group(1) == "TRUE"

def process_calc(history, response, debug: bool = False):
    regex = re.compile(r"```python\s*VIAJES\(\"(.*?)\"\)\s*```", re.DOTALL)
    matches = list(regex.finditer(response))
    final_response = response

    # Procesar solo el primer bloque VIAJES si existe
    if matches:
        match_str = matches[0].group(1)
        try:
            result = realizar_consulta(match_str, max_chunks=5, max_distance=0.9, debug=debug)
        except Exception:
            result = "Error al ejecutar la expresión"

        history.append(
            {
                "role": "user",
                "content": f'```python\nVIAJES("{match_str}") # resultado: {result}\n```',
            }
        )

        final_response = llm(history, debug)
        history.append({"role": "assistant", "content": final_response})

        # Obtener la pregunta original del usuario
        user_prompt = ""
        for message in history:
            if message["role"] == "user" and not message["content"].startswith("```python"):
                user_prompt = message["content"]
                break

        # Validar la respuesta y mejorarla si es necesario
        if not validar_respuesta(user_prompt, final_response, debug):
            try:
                dprint(f"** MEJORANDO RESPUESTA **: {final_response}", debug)
                result = realizar_consulta_mejorada(match_str, max_chunks=5, max_distance=0.9, responses=3, debug=debug)
            except Exception:
                result = "Error al ejecutar la expresión"

            history.append(
                {
                    "role": "user",
                    "content": f'```python\nVIAJES("{match_str}") # resultado: {result}\n```',
                }
            )

            final_response = llm(history, debug)
            history.append({"role": "assistant", "content": final_response})

    return final_response


if __name__ == "__main__":
    main()
