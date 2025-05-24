import openai

# TODO: Manejar el historico de la conversación
# TODO: Manejar la fecha, para que sepa en que día está
# TODO: guardar información del usuario para llamerle por su nombre
# TODO: guardar información del pais para utilizar el idioma 
# TODO: hacerlo con langchain despues de que esté funcionando
# El guardar el resultado un base de datos debe hacerse en otro modulo.
# Este modulo contiene exclusivamente la lógica del agente


def agent(history: list[dict]):
    llm = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history,
        temperature=0.7,
        max_tokens=150,
    )
