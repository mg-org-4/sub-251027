import imageio
import numpy as np
import torch
import subprocess
import os
import sys
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCKER_COMPOSE_FILE = os.path.join(BASE_DIR, "docker-compose-lite.yml")
CONTAINER_NAME = "heygem-gen-video"
VOLUME_ENV_VAR = "HEGEM_FACE2FACE_DATA_PATH"

os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'

def _run_docker_command(command_parts, check_result=False, capture_output=True, text_output=True, env=None):
    try:
        result = subprocess.run(
            command_parts,
            capture_output=capture_output,
            text=text_output,
            check=check_result,
            env=env
        )
        return result
    except FileNotFoundError:
        print(f"Error: '{command_parts[0]}' command not found.", file=sys.stderr)
        print(f"Please ensure Docker or Docker Compose is installed and in your system's PATH.", file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"Exit Code: {e.returncode}", file=sys.stderr)
        if e.stdout: print(f"Stdout:\n{e.stdout}", file=sys.stderr)
        if e.stderr: print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        raise

def is_docker_container_running(container_name):
    command = [
        "docker", "ps",
        "-q",
        "--filter", f"name={container_name}"
    ]
    try:
        result = _run_docker_command(command)
        if result.stdout.strip():
            print(f"Container '{container_name}' is running (ID: {result.stdout.strip()}).")
            return True
        else:
            return False
    except Exception:
        raise


def docker_container_exists(container_name):
    command = [
        "docker", "ps",
        "-aq",
        "--filter", f"name={container_name}"
    ]
    try:
        result = _run_docker_command(command)
        if result.stdout.strip():
            print(f"Container '{container_name}' exists (ID: {result.stdout.strip()}).")
            return True
        else:
            print(f"Container '{container_name}' does not exist.")
            return False
    except Exception:
        return False

def start_heygem_service(volume_host_path):
    if is_docker_container_running(CONTAINER_NAME):
        print(f"\nContainer '{CONTAINER_NAME}' is already running. No action needed.")
        return True

    if docker_container_exists(CONTAINER_NAME):
        start_command = ["docker", "start", CONTAINER_NAME]
        try:
            _run_docker_command(start_command, check_result=True)
            print(f"Successfully sent 'docker start {CONTAINER_NAME}' command.")
            
            time.sleep(5)
            if is_docker_container_running(CONTAINER_NAME):
                print(f"Container '{CONTAINER_NAME}' confirmed as running after direct start.")
                return True
            else:
                print(f"Container '{CONTAINER_NAME}' did not become ready after direct start.")
                return False

        except Exception:
            print(f"Failed to directly start container '{CONTAINER_NAME}'. Falling back to docker-compose.", file=sys.stderr)
            pass 

    docker_daemon_running = False
    try:
        _run_docker_command(["docker", "info"], check_result=True)
        docker_daemon_running = True
        print("Docker daemon is running and reachable.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("\n--- Docker Daemon Not Running ---", file=sys.stderr)
        print("Error: Could not connect to the Docker daemon.", file=sys.stderr)
        print("Cannot proceed with docker-compose without a running Docker daemon.", file=sys.stderr)
        print("\nPlease ensure the Docker daemon is running:", file=sys.stderr)

        if sys.platform.startswith('linux'):
            print("  - On Linux, try: 'sudo systemctl start docker' or 'sudo service docker start'", file=sys.stderr)
        elif sys.platform == 'win32':
             print("  - On Windows with Docker Desktop: Ensure Docker Desktop application is running.", file=sys.stderr)
             print("  - If running within WSL: Ensure Docker Desktop on Windows is running and WSL integration is enabled.", file=sys.stderr)
        elif sys.platform == 'darwin':
             print("  - On macOS: Ensure Docker Desktop application is running.", file=sys.stderr)
        else:
             print("  - Please consult your operating system documentation for starting the Docker service.", file=sys.stderr)

        print("-----------------------------------", file=sys.stderr)
        raise

    if docker_daemon_running:
        env_for_subprocess = os.environ.copy()
        env_for_subprocess[VOLUME_ENV_VAR] = volume_host_path
        env_for_subprocess['PWD'] = os.getcwd()

        compose_up_command = [
            "docker-compose",
            "-f",
            DOCKER_COMPOSE_FILE,
            "up",
            "-d"
        ]

        try:
            subprocess.run(
                compose_up_command,
                check=True,
                env=env_for_subprocess,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            print(f"\nSuccessfully executed 'docker-compose up -d' for '{CONTAINER_NAME}'.")

            wait_sec = 0
            max_wait_sec = 60
            wait_interval_sec = 3

            while wait_sec < max_wait_sec:
                if is_docker_container_running(CONTAINER_NAME):
                    print(f"Container '{CONTAINER_NAME}' confirmed as running after docker-compose up.")
                    return True

                try:
                    status_command = ["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER_NAME]
                    status_result = _run_docker_command(status_command)
                    status = status_result.stdout.strip()
                except Exception:
                    pass

                time.sleep(wait_interval_sec)
                wait_sec += wait_interval_sec

            print(f"\nError: Timed out waiting for container '{CONTAINER_NAME}' to start after docker-compose up ({max_wait_sec}s).", file=sys.stderr)
            print("Please check container logs for details (e.g., 'docker logs {}').".format(CONTAINER_NAME), file=sys.stderr)
            return False

        except Exception:
            print(f"\nFailed to start '{CONTAINER_NAME}' using docker-compose. The command output above should provide details.", file=sys.stderr)
            return False

    else:
         print("Skipping docker-compose up because Docker daemon is not running.", file=sys.stderr)
         return False

def _get_container_id(container_name):
    command = [
        "docker", "ps",
        "-q",
        "--filter", f"name={container_name}"
    ]
    try:
        result = _run_docker_command(command)
        container_id = result.stdout.strip()
        if container_id:
            return container_id
        return None
    except Exception as e:
        print(f"Error getting container ID for '{container_name}': {e}", file=sys.stderr)
        return None
       
def stop_heygem_service():
    current_container_id = _get_container_id(CONTAINER_NAME)
    if not current_container_id:
        print(f"\nContainer '{CONTAINER_NAME}' is not running. No need to stop.")
        return True

    stop_command = ["docker", "stop", current_container_id]

    try:
        _run_docker_command(stop_command, check_result=True)
        print(f"\nSuccessfully sent 'docker stop {CONTAINER_NAME}' command.")

    except Exception as e:
        print(f"\nFailed to stop '{CONTAINER_NAME}' using 'docker stop': {e}. Please check the logs above.", file=sys.stderr)
        
def process_tensor_by_duration(image_tensor, duration, mode='normal', fps=24):
    original_frames_count = image_tensor.shape[0]
    target_frames_count = int(duration * fps)
    
    if mode == 'normal':
        if target_frames_count <= original_frames_count:
            return image_tensor[:target_frames_count]
        else:
            return image_tensor
    
    frame_indices_to_write = []
    original_indices = np.arange(original_frames_count)
    
    if mode == 'repeat':
        if target_frames_count <= original_frames_count:
            return image_tensor[:target_frames_count]
        
        while len(frame_indices_to_write) < target_frames_count:
            frame_indices_to_write.extend(original_indices)
        frame_indices_to_write = frame_indices_to_write[:target_frames_count]
        
    elif mode == 'pingpong':
        if target_frames_count <= original_frames_count:
            return image_tensor[:target_frames_count]
            
        forward_indices = original_indices
        backward_indices = original_indices[::-1]
        
        while len(frame_indices_to_write) < target_frames_count:
            frame_indices_to_write.extend(forward_indices)
            if len(frame_indices_to_write) < target_frames_count:
                frame_indices_to_write.extend(backward_indices)
        frame_indices_to_write = frame_indices_to_write[:target_frames_count]
    
    return image_tensor[frame_indices_to_write]

def save_tensor_as_video(image_tensor, output_path, fps=24):
    if image_tensor.is_cuda:
        tensor_np = image_tensor.cpu().numpy()
    else:
        tensor_np = image_tensor.numpy()

    tensor_np = np.clip(np.round(tensor_np * 255), 0, 255).astype(np.uint8)

    try:
        writer_params = {'codec': 'ffv1'}
        with imageio.get_writer(output_path, fps=fps, **writer_params) as writer:
            for frame in tensor_np:
                writer.append_data(frame)
        print(f"Successfully saved video to {output_path} using FFV1 (lossless).")

    except Exception as e:
        print(f"Failed to use FFV1 codec: {e}. Trying libx264 with crf=0 (lossless, but might have YUV conversion).")
        try:
            ffmpeg_params = ['-crf', '0', '-pix_fmt', 'yuv444p']
            
            with imageio.get_writer(output_path, fps=fps, codec='libx264', ffmpeg_params=ffmpeg_params) as writer:
                for frame in tensor_np:
                    writer.append_data(frame)

            print(f"Successfully saved video to {output_path} using libx264 (crf=0, lossless).")

        except Exception as e_h264:
            print(f"Failed to use libx264 crf=0: {e_h264}. Falling back to default (potentially lossy).")
            try:
                with imageio.get_writer(output_path, fps=fps) as writer:
                    for frame in tensor_np:
                        writer.append_data(frame)
                print(f"Saved video to {output_path} using default imageio settings (potentially lossy).")
            except Exception as e_default:
                print(f"Failed even with default settings: {e_default}")
                print("Video saving failed.")

def video_to_tensor(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    reader = None
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()

        width, height = meta['size']
        channels = 3

        def frame_processor_generator(imgio_reader, expected_shape):
            for i, frame in enumerate(imgio_reader):
                if frame.shape[-1] == 4 and expected_shape[-1] == 3:
                     frame = frame[:, :, :3]
                if frame.shape != expected_shape:
                    print(f"Warning: Frame {i} has unexpected shape {frame.shape}. Expected {expected_shape}. Skipping.")
                    continue

                frame_processed = (frame.astype(np.float32) / 255.0)

                yield frame_processed

        frame_shape = (height, width, channels)
        gen_instance = frame_processor_generator(reader, frame_shape)

        frames_np_flat = np.fromiter(
            gen_instance,
            np.dtype((np.float32, frame_shape))
        )

        num_loaded_frames = len(frames_np_flat)

        if num_loaded_frames == 0:
            reader.close()
            raise RuntimeError(f"Failed to load any frames from video '{video_path}'. Check video file.")

        frame_shape = (height, width, channels)
        gen_instance = frame_processor_generator(reader, frame_shape)

        frames_structured_np = np.fromiter(
            gen_instance,
            dtype=np.dtype((np.float32, frame_shape))
        )
        print(f"Finished reading frames. Structured numpy array shape: {frames_structured_np.shape}, dtype: {frames_structured_np.dtype}")

        num_loaded_frames = len(frames_structured_np)

        if num_loaded_frames == 0:
            reader.close()
            raise RuntimeError(f"Failed to load any frames from video '{video_path}'. Check video file or its content.")

        total_scalars = num_loaded_frames * height * width * channels
        try:
            if frames_structured_np.size * frames_structured_np.dtype.itemsize != total_scalars * np.dtype(np.float32).itemsize:
                 pass

            frames_np = frames_structured_np.view(np.float32).reshape(-1, height, width, channels)
            print(f"Reshaped numpy array shape: {frames_np.shape}, dtype: {frames_np.dtype}")

        except Exception as reshape_e:
            print(f"Error reshaping numpy array after fromiter: {reshape_e}")
            reader.close()
            raise RuntimeError(f"Failed to reshape frame data loaded from '{video_path}'. Likely mismatch in expected frame dimensions or data.") from reshape_e

        try:
            tensor = torch.from_numpy(frames_np)
            print("Tensor conversion successful.")
        except Exception as tensor_e:
            print(f"Error converting numpy array to torch tensor: {tensor_e}")
            reader.close()
            raise RuntimeError(f"Failed to convert numpy array to torch tensor for '{video_path}'. Likely out of memory.") from tensor_e

        reader.close()

        print(f"Successfully loaded video '{video_path}' into tensor with shape {tensor.shape}.")
        return tensor

    except Exception as e:
        print(f"An error occurred during video loading for '{video_path}': {e}")
        if reader is not None:
             try:
                 reader.close()
             except Exception:
                 pass
        raise