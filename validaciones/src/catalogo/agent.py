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
    process_agent(history, debug)
    return history

def process_agent(history, debug: bool = False):
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

def validar_respuesta(prompt: str, response: str, debug: bool = False) -> bool:
    prompt = validate_prompt.format(prompt=prompt, response=response)
    response = llm([{"role": "user", "content": prompt}], debug)
    regex = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    matches = list(regex.finditer(response))
    return matches[0].group(1) == "TRUE"

def obtener_pregunta_usuario(history):
    """
    Extrae la pregunta original del usuario del historial.
    """
    for message in history:
        if message["role"] == "user" and not message["content"].startswith("```python"):
            return message["content"]
    return ""

def extraer_llamadas_viajes(response):
    """
    Extrae las llamadas a la función VIAJES de la respuesta.
    """
    regex = re.compile(r"```python\s*VIAJES\(\s*\"(.*?)\"\s*\)\s*```", re.DOTALL)
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

def añadir_resultado_al_historial(history, consulta, resultado, es_mejorado=False):
    """
    Añade el resultado de una consulta al historial.
    """
    tipo = "mejorado" if es_mejorado else ""
    history.append(
        {
            "role": "user",
            "content": f'```python\nVIAJES("{consulta}") # resultado {tipo}: {resultado}\n```',
        }
    )

def procesar_respuesta_llm(history, debug=False):
    """
    Procesa la respuesta del LLM y elimina bloques de código VIAJES si es necesario.
    """
    try:
        final_response = llm(history, debug)
        if "```python" in final_response and "VIAJES" in final_response:
            dprint("La respuesta aún contiene bloques de código VIAJES", debug)
            text_parts = final_response.split("```python")
            if text_parts and text_parts[0].strip():
                return text_parts[0].strip()
            else:
                # Extraer la consulta original del historial
                consulta = ""
                for msg in history:
                    if msg["role"] == "user" and "VIAJES" in msg["content"]:
                        match = re.search(r'VIAJES\("(.*?)"\)', msg["content"])
                        if match:
                            consulta = match.group(1)
                            break
                return f"Basado en tu consulta sobre '{consulta}', no pude encontrar información específica."
        return final_response
    except Exception as e:
        dprint(f"Error al generar la respuesta: {str(e)}", debug)
        # Extraer la consulta original del historial
        consulta = ""
        for msg in history:
            if msg["role"] == "user" and "VIAJES" in msg["content"]:
                match = re.search(r'VIAJES\("(.*?)"\)', msg["content"])
                if match:
                    consulta = match.group(1)
                    break
        return f"Basado en tu consulta sobre '{consulta}', no pude encontrar información específica."

def actualizar_historial_con_respuesta(history, respuesta):
    """
    Actualiza el historial con la respuesta final.
    """
    if history[-1]["role"] == "assistant":
        history[-1]["content"] = respuesta
    else:
        history.append({"role": "assistant", "content": respuesta})

def process_calc(history, response, debug: bool = False):
    """
    Procesa la respuesta del LLM y realiza consultas a la base de datos de viajes si es necesario.
    """
    # Inicialización
    final_response = response
    user_prompt = obtener_pregunta_usuario(history)
    matches = extraer_llamadas_viajes(response)
    
    # Procesar llamadas a VIAJES si existen
    if matches:
        match_str = matches[0].group(1)
        dprint(f"Consulta extraída: {match_str}", debug)
        
        # Ejecutar consulta
        result = ejecutar_consulta_viajes(match_str, debug)
        
        # Añadir resultado al historial
        añadir_resultado_al_historial(history, match_str, result)
        
        # Procesar respuesta del LLM
        final_response = procesar_respuesta_llm(history, debug)
        
        # Actualizar historial
        actualizar_historial_con_respuesta(history, final_response)
        
        consulta_validar = match_str
    else:
        # Si no hay llamada a VIAJES, usamos la pregunta original para validar
        consulta_validar = user_prompt

    # Validar y mejorar la respuesta si es necesario
    try:
        is_valid = validar_respuesta(user_prompt, final_response, debug)
        if not is_valid:
            dprint(f"** MEJORANDO RESPUESTA **: {final_response}", debug)
            
            # Ejecutar consulta mejorada
            result = ejecutar_consulta_mejorada_viajes(consulta_validar, debug)
            
            # Añadir resultado mejorado al historial
            añadir_resultado_al_historial(history, consulta_validar, result, es_mejorado=True)
            
            # Procesar respuesta mejorada del LLM
            final_response = procesar_respuesta_llm(history, debug)
            
            # Actualizar historial con la respuesta mejorada
            actualizar_historial_con_respuesta(history, final_response)
    except Exception as e:
        dprint(f"Error en la validación: {str(e)}", debug)
        # No modificar final_response si hay un error en la validación

    return final_response


if __name__ == "__main__":
    main()
