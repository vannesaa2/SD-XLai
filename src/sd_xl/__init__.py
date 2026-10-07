import torch
from diffusers import AutoPipelineForText2Image

# Diffusers es una libreria especializada en modelos generativos
print("Cargar el modelo...")

# Corregido el nombre del modelo y eliminados los espacios al inicio
modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32
    # torch.floatXX indica la precisión numérica con la que carga los pesos
)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear: ")
# negative_prompt = "(peor calidad, baja resolución:1.4), (imagen borrosa, desenfoque:1.2), artefactos de compresión, anatomía defectuosa, extremidades deformadas, 
print("imagen en curso....")

imagen = modelo(
    prompt=prompt, # Instrucciones o detalles para crear la imagen
 #   negative_prompt= negative_prompt,
    num_inference_steps=25, # Las veces que va a tratar de crear o refinar la imagen
    guidance_scale=7.0, # <-- Se agregó la coma que faltaba aquí
    height=1024,
    width=1024
).images[0]

imagen.save("imagen.png")
print("imagen Guardada")

