import re
import json
import time
import random
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv
from catalogo.agent import run_agent

load_dotenv()

langfuse = Langfuse()

def get_malsonantes():
    archivo = "src/tests/malsonantes.txt"
    with open(archivo, "r") as file:
        return file.read().splitlines()

def malsonantes_eval(history, _):
    malsonantes = get_malsonantes()
    output = history[-1]["content"]
    count = 0
    for malsonante in malsonantes:
        if malsonante not in output.lower():
            count += 1
    return count / len(malsonantes)

def tool_usage_eval(history, expected_tool_call):
    # Para test_tool_usage verificamos si la herramienta esperada está en la respuesta
    tool_call = history[-2]["content"].lower()
    
    if isinstance(expected_tool_call, str):
        # Construir patrón de búsqueda con el término esperado
        expected_pattern = f"""rag_viajes\\("[^"]*{expected_tool_call.lower()}[^"]*"\\)"""
        if re.search(expected_pattern, tool_call):
            return 1.0  # Éxito: se encontró la herramienta esperada
    
    return 0.0  # Fallo: no se encontró la herramienta esperada

# Función para ejecutar el agente con reintentos en caso de rate limit
def run_agent_with_retry(input_text, max_retries=5, initial_wait=2):
    retries = 0
    while retries <= max_retries:
        try:
            # Pequeña pausa antes de cada ejecución para evitar rate limits
            time.sleep(0.5)
            # Ejecutar el agente
            return run_agent(input_text)
        except Exception as e:
            error_str = str(e)
            # Verificar si es un error de rate limit
            if "rate limit" in error_str.lower() or "429" in error_str:
                retries += 1
                if retries > max_retries:
                    raise Exception(f"Máximo de reintentos alcanzado: {error_str}")
                
                # Tiempo de espera exponencial con jitter
                wait_time = initial_wait * (2 ** (retries - 1)) + random.uniform(0, 1)
                print(f"Rate limit alcanzado. Reintentando en {wait_time:.2f} segundos...")
                time.sleep(wait_time)
            else:
                # Si no es un error de rate limit, propagar la excepción
                raise

# Diccionario de funciones de evaluación por tipo de experimento
test_functions = {
   "malsonantes": malsonantes_eval,
   "tool_usage": tool_usage_eval
}

def run_experiments():
    # Obtener el dataset unificado
    dataset = langfuse.get_dataset("evaluaciones")
    
    for item in dataset.items:
        try:
            # Manejar el caso donde item.input puede ser diccionario o string
            if isinstance(item.input, dict):
                data = item.input
            else:
                # Intentar parsear como JSON si es string
                data = json.loads(item.input)
                
            experiment_name = data.get("experiment")
            real_input = data.get("input")
            expected_output = item.expected_output
            
            if not experiment_name or not real_input:
                print(f"Error: formato incorrecto en item {item.id}")
                continue
                
            # Verificar que existe una función de evaluación para este experimento
            if experiment_name not in test_functions:
                print(f"Error: no existe función de evaluación para {experiment_name}")
                continue
            
            # Obtener la función de evaluación correspondiente
            test_function = test_functions[experiment_name]
            
            # Ejecutar la prueba con observación y etiquetado
            with item.observe(run_name=experiment_name) as trace_id:
                # Ejecutar el agente con el input real usando la función con reintentos
                output = run_agent_with_retry(real_input)
                
                # Evaluar y registrar el resultado
                langfuse.score(
                    trace_id=trace_id,
                    name=experiment_name,
                    value=test_function(output, expected_output)
                )
                
        except json.JSONDecodeError:
            print(f"Error: el input no es un JSON válido en item {item.id}")
        except Exception as e:
            print(f"Error procesando item {item.id}: {str(e)}")

def main():
    run_experiments()

if __name__ == "__main__":
    main()