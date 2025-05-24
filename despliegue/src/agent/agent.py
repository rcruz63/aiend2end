import openai

# TODO: Manejar el historico de la conversación
# TODO: Manejar la fecha, para que sepa en que día está
# TODO: guardar información del usuario para llamerle por su nombre
# TODO: guardar información del pais para utilizar el idioma
# TODO: hacerlo con langchain despues de que esté funcionando
# TODO: recibir convertsation_id en lugar de history y recuperar desde la memoria
# El guardar el resultado un base de datos debe hacerse en otro modulo.
# Este modulo contiene exclusivamente la lógica del agente

SYSTEM_PROMPT = """
Eres un asistente de IA que ayuda a aprender ingles a través de las conversaciones.
Habla de forma natural y amigable.

Intenta detectar los errores que comete el usuario y corregirlos. Prpon ejercicios para practicar.
"""


def agent(history: list[dict]) -> str:

    full_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ] + history
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=full_history,
    )
    return response.choices[0].message.content
