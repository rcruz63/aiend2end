import pytest
from catalogo.agent import run_agent
import os

def test_agent():
    assert True

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
    for malsonante in malsonantes:
        assert malsonante not in last_message.lower()

@pytest.mark.full
@pytest.mark.parametrize("conversacion", get_conversaciones(rapido=False))
def test_malsonantes_completo(conversacion):
    malsonantes = get_malsonantes(rapido=False)
    prueba_unitaria(conversacion, malsonantes)

@pytest.mark.fast
@pytest.mark.parametrize("conversacion", get_conversaciones(rapido=True))
def test_malsonantes(conversacion):
    malsonantes = get_malsonantes(rapido=False)
    prueba_unitaria(conversacion, malsonantes)

# Apunte: Un test puede ser comprobar el formato, por ejemplo, si pides un json, o si pides solo un "SI" o "NO"