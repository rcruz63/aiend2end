import pytest
from pprint import pprint
import re
from catalogo.agent import run_agent
import os


def get_malsonantes(rapido=True):
    archivo = "src/tests/malsonantes_rapidas.txt" if rapido else "src/tests/malsonantes.txt"
    with open(archivo, "r") as file:
        return file.read().splitlines()

def get_conversaciones(rapido=True):
    archivo = "src/tests/frases_iniciales_rapidas.txt" if rapido else "src/tests/frases_iniciales.txt"
    with open(archivo, "r") as file:
        return file.read().splitlines()
    
def prueba_unitaria(conversacion: str, malsonantes: list[str]):
    history = run_agent(conversacion)
    last_message = history[-1]["content"]
    print(f"Última respuesta: {last_message}")
    for malsonante in malsonantes:
        assert malsonante not in last_message.lower()

# @pytest.mark.parametrize("conversacion", get_conversaciones(rapido=False))
# def test_malsonantes_completo(conversacion):
#     malsonantes = get_malsonantes(rapido=False)
#     prueba_unitaria(conversacion, malsonantes)

@pytest.mark.parametrize("conversacion", get_conversaciones(rapido=True))
def test_malsonantes(conversacion):
    malsonantes = get_malsonantes(rapido=False)
    prueba_unitaria(conversacion, malsonantes)

with open("src/tests/positive_tool_calls.csv", "r") as file:
    lines = file.read().splitlines()
    positive_tool_calls = [line.split(",") for line in lines]

@pytest.mark.parametrize("conversacion", positive_tool_calls)
def test_tool_usage(conversacion):
    input, expected_tool_call = conversacion
    history = run_agent(input)
    tool_call = history[-2]["content"].lower()
    assert re.search(expected_tool_call, tool_call)

with open("src/tests/negative_tool_calls.txt", "r") as file:
    negative_tool_calls = file.read().splitlines()

@pytest.mark.parametrize("conversacion", negative_tool_calls)
def test_tool_usage_negative(conversacion):
    history = run_agent(conversacion)
    tool_call = history[-2]["content"].lower()
    assert 'rag_viajes("' not in tool_call

# Apunte: Un test puede ser comprobar el formato, por ejemplo, si pides un json, o si pides solo un "SI" o "NO"