import os
import re
import shutil
import cairosvg
import subprocess
from collections import defaultdict
from tqdm import tqdm

# Configuración
svg_dir = "renders"
png_root = "pngs"
video_dir = "videos"
frame_rate = 5  # Frames por segundo
frame_duration = 1/frame_rate  # Duración de cada frame en segundos

# Crear directorios
os.makedirs(png_root, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Patrones para identificar archivos
pattern = re.compile(r'(\w+)-map(\d+)-agent(\d+)-epoch(\d+)\.svg')
groups = defaultdict(list)

# Procesar archivos SVG
for filename in sorted(os.listdir(svg_dir)):
    if filename.endswith(".svg"):
        match = pattern.match(filename)
        if match:
            concept, map_id, agent_id, epoch = match.groups()
            key = f"{concept}_map{map_id}_agent{agent_id}"
            groups[key].append((int(epoch), filename))

# Procesar cada grupo
for key, files in groups.items():
    # Ordenar archivos por epoch
    files.sort(key=lambda x: x[0])
    png_dir = os.path.join(png_root, key)
    os.makedirs(png_dir, exist_ok=True)

    print(f"[{key}] Procesando {len(files)} frames...")

    # Convertir SVGs a PNGs con nombres secuenciales
    for i, (epoch, svg_file) in enumerate(tqdm(files)):
        input_path = os.path.join(svg_dir, svg_file)
        output_path = os.path.join(png_dir, f"frame_{i:05d}.png")
        cairosvg.svg2png(url=input_path, write_to=output_path)

    # Verificar que todos los frames se hayan generado
    generated_frames = sorted(os.listdir(png_dir))
    expected_frames = [f"frame_{i:05d}.png" for i in range(len(files))]

    if generated_frames != expected_frames:
        print(f"⚠️ Error: Frames generados no coinciden con los esperados para {key}")
        print(f"Esperados: {expected_frames}")
        print(f"Generados: {generated_frames}")
        continue

    output_video = os.path.join(video_dir, f"{key}.mp4")
    print(f"[{key}] Generando video...")

    # Crear archivo de texto con la lista de frames para ffmpeg
    frame_list_path = os.path.join(png_dir, "frames.txt")
    with open(frame_list_path, 'w') as f:
        for frame in expected_frames:
            f.write(f"file 'frame_{frame.split('_')[1]}'\n")
            f.write(f"duration {frame_duration}\n")

    # Usar ffmpeg con el archivo de lista para asegurar el orden y duración correctos
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Sobrescribir sin preguntar
        "-f", "concat",
        "-safe", "0",
        "-i", frame_list_path,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_video
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"[{key}] ✅ Video generado correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al generar video para {key}: {e}")
        continue

    # Eliminar PNGs temporales
    try:
        #shutil.rmtree(png_dir)
        print(f"[{key}] ✅ PNGs temporales eliminados.")
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudieron eliminar los PNGs temporales para {key}: {e}")

print("🎬 Todos los videos han sido generados correctamente.")
