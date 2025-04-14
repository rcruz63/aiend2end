import re
from pprint import pprint

import click
from openai import OpenAI
from .rag import realizar_consulta, realizar_consulta_mejorada
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
Un RAG sobre catálogos de viajes y ofertas sobre viajes ha generado una respuesta a una pregunta del usuario.

La respuesta recibida ha sido:

{response}

La pregunta del usuario ha sido:

{prompt}

Por favor, verifica que la respuesta es coherente con la pregunta y proporciona información útil sobre viajes.

Si la respuesta es válida y proporciona información específica sobre viajes relacionada con la pregunta, responde:

```python
TRUE
```

Si la respuesta no es válida por alguna de estas razones:
1. Contiene todavía bloques de código con llamadas a VIAJES
2. Dice "Lo siento, pero no tengo información sobre ofertas de viajes" o algo similar
3. No responde directamente a la pregunta del usuario sobre viajes
4. Es muy genérica y no proporciona información específica sobre destinos, itinerarios o detalles de viajes

Entonces responde:

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
    regex = re.compile(r"```python\s*VIAJES\(\s*\"(.*?)\"\s*\)\s*```", re.DOTALL)
    matches = list(regex.finditer(response))
    final_response = response
    
    # Obtener la pregunta original del usuario
    user_prompt = ""
    for message in history:
        if message["role"] == "user" and not message["content"].startswith("```python"):
            user_prompt = message["content"]
            break

    # Si hay una llamada a VIAJES, procesarla
    if matches:
        match_str = matches[0].group(1)
        dprint(f"Consulta extraída: {match_str}", debug)
        try:
            result = realizar_consulta(match_str, max_chunks=5, max_distance=0.9, debug=debug)
            if not result or result.strip() == "":
                result = "No se encontró información específica sobre esta consulta en nuestra base de datos de viajes."
        except Exception as e:
            dprint(f"Error al ejecutar la consulta: {str(e)}", debug)
            result = "Error al procesar la consulta. Por favor, inténtalo de nuevo con una pregunta más específica sobre viajes."

        # Añadir el resultado al historial
        history.append(
            {
                "role": "user",
                "content": f'```python\nVIAJES("{match_str}") # resultado: {result}\n```',
            }
        )

        try:
            final_response = llm(history, debug)
            # Verificar si la respuesta sigue conteniendo bloques de código VIAJES
            if "```python" in final_response and "VIAJES" in final_response:
                dprint("La respuesta aún contiene bloques de código VIAJES", debug)
                # Intentar extraer solo la parte de texto antes del bloque de código
                text_parts = final_response.split("```python")
                if text_parts and text_parts[0].strip():
                    final_response = text_parts[0].strip()
                else:
                    final_response = f"Basado en tu consulta sobre '{match_str}', {result}"
        except Exception as e:
            dprint(f"Error al generar la respuesta final: {str(e)}", debug)
            final_response = f"Basado en tu consulta sobre '{match_str}', {result}"

        history.append({"role": "assistant", "content": final_response})
        
        consulta_validar = match_str
    else:
        # Si no hay llamada a VIAJES, usamos la pregunta original para validar
        consulta_validar = user_prompt

    # Validar la respuesta y mejorarla si es necesario
    try:
        is_valid = validar_respuesta(user_prompt, final_response, debug)
        if not is_valid:
            dprint(f"** MEJORANDO RESPUESTA **: {final_response}", debug)
            try:
                result = realizar_consulta_mejorada(consulta_validar, max_chunks=5, max_distance=0.9, responses=3, debug=debug)
                if not result or result.strip() == "":
                    result = "No se encontró información específica sobre esta consulta en nuestra base de datos de viajes."
            except Exception as e:
                dprint(f"Error al ejecutar la consulta mejorada: {str(e)}", debug)
                result = "No se encontró información específica sobre esta consulta en nuestra base de datos de viajes."

            history.append(
                {
                    "role": "user",
                    "content": f'```python\nVIAJES("{consulta_validar}") # resultado mejorado: {result}\n```',
                }
            )

            try:
                final_response = llm(history, debug)
                # Verificar nuevamente si la respuesta contiene bloques de código
                if "```python" in final_response and "VIAJES" in final_response:
                    text_parts = final_response.split("```python")
                    if text_parts and text_parts[0].strip():
                        final_response = text_parts[0].strip()
                    else:
                        final_response = f"Basado en tu consulta sobre '{consulta_validar}', {result}"
            except Exception as e:
                dprint(f"Error al generar la respuesta mejorada: {str(e)}", debug)
                final_response = f"Basado en tu consulta sobre '{consulta_validar}', {result}"

            history.append({"role": "assistant", "content": final_response})
    except Exception as e:
        dprint(f"Error en la validación: {str(e)}", debug)
        # No modificar final_response si hay un error en la validación

    return final_response


if __name__ == "__main__":
    main()
