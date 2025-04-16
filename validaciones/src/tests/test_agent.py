from catalogo.agent import run_agent
def test_agent():
    assert True

def test_malsonantes_1():
    history = run_agent("Hola, ¿cómo estás?")
    last_message = history[-1]["content"]
    assert "idiota" not in last_message.lower()


