import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import json
import asyncio
import aiohttp
from aiohttp import web
import io
import server
import comfy.model_management
import subprocess
import platform
import time
import atexit
import signal
import sys
import shlex
import uuid
from multiprocessing import Queue
from comfy.utils import ProgressBar

# Import shared utilities
from .utils.logging import debug_log, log
from .utils.config import (
    CONFIG_FILE,
    get_default_config,
    load_config,
    save_config,
    ensure_config_exists,
    get_worker_timeout_seconds,
    is_master_delegate_only,
)
from .utils.image import tensor_to_pil, pil_to_tensor, ensure_contiguous
from .utils.process import is_process_alive, terminate_process, get_python_executable
from .utils.network import handle_api_error, get_server_port, get_server_loop, get_client_session, cleanup_client_session
from .utils.async_helpers import run_async_in_server_loop
from .utils.constants import (
    WORKER_JOB_TIMEOUT, PROCESS_TERMINATION_TIMEOUT, WORKER_CHECK_INTERVAL, 
    STATUS_CHECK_INTERVAL, CHUNK_SIZE, LOG_TAIL_BYTES, WORKER_LOG_PATTERN, 
    WORKER_STARTUP_DELAY, PROCESS_WAIT_TIMEOUT, MEMORY_CLEAR_DELAY, MAX_BATCH
)

# Try to import psutil for better process management
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    log("psutil not available, using fallback process management")
    PSUTIL_AVAILABLE = False

# Register cleanup for aiohttp session
def cleanup():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(cleanup_client_session())
    loop.close()

atexit.register(cleanup)

# --- API Endpoints ---
@server.PromptServer.instance.routes.get("/distributed/config")
async def get_config_endpoint(request):
    config = load_config()
    return web.json_response(config)

@server.PromptServer.instance.routes.get("/distributed/queue_status/{job_id}")
async def queue_status_endpoint(request):
    """Check if a job queue is initialized."""
    try:
        job_id = request.match_info['job_id']
        
        # Import to ensure initialization
        from .distributed_upscale import ensure_tile_jobs_initialized
        prompt_server = ensure_tile_jobs_initialized()
        
        async with prompt_server.distributed_tile_jobs_lock:
            exists = job_id in prompt_server.distributed_pending_tile_jobs
        
        debug_log(f"Queue status check for job {job_id}: {'exists' if exists else 'not found'}")
        return web.json_response({"exists": exists, "job_id": job_id})
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.post("/distributed/worker/clear_launching")
async def clear_launching_state(request):
    """Clear the launching flag when worker is confirmed running."""
    try:
        data = await request.json()
        worker_id = str(data.get('worker_id'))
        
        if not worker_id:
            return await handle_api_error(request, "worker_id is required", 400)
        
        # Clear launching flag in managed processes
        if worker_id in worker_manager.processes:
            if 'launching' in worker_manager.processes[worker_id]:
                del worker_manager.processes[worker_id]['launching']
                worker_manager.save_processes()
                debug_log(f"Cleared launching state for worker {worker_id}")
        
        return web.json_response({"status": "success"})
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.get("/distributed/network_info")
async def get_network_info_endpoint(request):
    """Get network interfaces and recommend best IP for master."""
    import socket
    
    # Get CUDA device if available
    cuda_device = None
    cuda_device_count = 0
    physical_device_count = 0
    
    if torch.cuda.is_available():
        try:
            import os
            import subprocess
            
            # Get visible device count (what PyTorch sees)
            cuda_device_count = torch.cuda.device_count()
            
            # Try to get actual physical device info
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            
            # Method 1: Parse CUDA_VISIBLE_DEVICES
            if cuda_visible and cuda_visible.strip():
                visible_devices = [int(d.strip()) for d in cuda_visible.split(',') if d.strip().isdigit()]
                if visible_devices:
                    # Get the first visible device as the actual physical device
                    cuda_device = visible_devices[0]
                    
                    # Try to get total physical device count using nvidia-smi
                    try:
                        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            physical_device_count = len(result.stdout.strip().split('\n'))
                        else:
                            physical_device_count = max(visible_devices) + 1  # Best guess
                    except:
                        physical_device_count = max(visible_devices) + 1  # Best guess
                else:
                    cuda_device = 0
                    physical_device_count = cuda_device_count
            else:
                # No CUDA_VISIBLE_DEVICES set, current device is actual device
                cuda_device = torch.cuda.current_device()
                physical_device_count = cuda_device_count
                
        except Exception as e:
            debug_log(f"CUDA detection error: {e}")
            cuda_device = None
            cuda_device_count = 0
            physical_device_count = 0
    
    def get_network_ips():
        """Get all network IPs, trying multiple methods."""
        ips = []
        hostname = socket.gethostname()
        
        # Method 1: Try socket.getaddrinfo
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for info in addr_info:
                ip = info[4][0]
                if ip and ip not in ips and not ip.startswith('::'):  # Skip IPv6 for now
                    ips.append(ip)
        except:
            pass
        
        # Method 2: Try to connect to external server and get local IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # Google DNS
            local_ip = s.getsockname()[0]
            s.close()
            if local_ip not in ips:
                ips.append(local_ip)
        except:
            pass
        
        # Method 3: Platform-specific commands
        
        try:
            if platform.system() == "Windows":
                # Windows ipconfig
                result = subprocess.run(["ipconfig"], capture_output=True, text=True)
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if 'IPv4' in line and i + 1 < len(lines):
                        ip = lines[i].split(':')[-1].strip()
                        if ip and ip not in ips:
                            ips.append(ip)
            else:
                # Unix/Linux/Mac ifconfig or ip addr
                try:
                    result = subprocess.run(["ip", "addr"], capture_output=True, text=True)
                except:
                    result = subprocess.run(["ifconfig"], capture_output=True, text=True)
                
                import re
                ip_pattern = re.compile(r'inet\s+(\d+\.\d+\.\d+\.\d+)')
                for match in ip_pattern.finditer(result.stdout):
                    ip = match.group(1)
                    if ip and ip not in ips:
                        ips.append(ip)
        except:
            pass
        
        return ips
    
    def get_recommended_ip(ips):
        """Choose the best IP for master-worker communication."""
        # Priority order:
        # 1. Private network ranges (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
        # 2. Other non-localhost IPs
        # 3. Localhost as last resort
        
        private_ips = []
        public_ips = []
        
        for ip in ips:
            if ip.startswith('127.') or ip == 'localhost':
                continue
            elif (ip.startswith('192.168.') or 
                  ip.startswith('10.') or 
                  (ip.startswith('172.') and 16 <= int(ip.split('.')[1]) <= 31)):
                private_ips.append(ip)
            else:
                public_ips.append(ip)
        
        # Prefer private IPs
        if private_ips:
            # Prefer 192.168 range as it's most common
            for ip in private_ips:
                if ip.startswith('192.168.'):
                    return ip
            return private_ips[0]
        elif public_ips:
            return public_ips[0]
        elif ips:
            return ips[0]
        else:
            return None
    
    try:
        hostname = socket.gethostname()
        all_ips = get_network_ips()
        recommended_ip = get_recommended_ip(all_ips)
        
        return web.json_response({
            "status": "success",
            "hostname": hostname,
            "all_ips": all_ips,
            "recommended_ip": recommended_ip,
            "cuda_device": cuda_device,
            "cuda_device_count": physical_device_count if physical_device_count > 0 else cuda_device_count,
            "message": "Auto-detected network configuration"
        })
    except Exception as e:
        return web.json_response({
            "status": "error",
            "message": str(e),
            "hostname": "unknown",
            "all_ips": [],
            "recommended_ip": None
        })

@server.PromptServer.instance.routes.get("/distributed/system_info")
async def get_system_info_endpoint(request):
    """Get system information including machine ID for local worker detection."""
    try:
        import socket
        
        return web.json_response({
            "status": "success",
            "hostname": socket.gethostname(),
            "machine_id": get_machine_id(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "node": platform.node(),
                "path_separator": os.sep,  # Add path separator
                "os_name": os.name  # Add OS name (posix, nt, etc.)
            },
            "is_docker": is_docker_environment(),
            "is_runpod": is_runpod_environment(),
            "runpod_pod_id": os.environ.get('RUNPOD_POD_ID')
        })
    except Exception as e:
        return web.json_response({
            "status": "error",
            "message": str(e)
        }, status=500)

@server.PromptServer.instance.routes.post("/distributed/config/update_worker")
async def update_worker_endpoint(request):
    try:
        data = await request.json()
        worker_id = data.get("worker_id")
        
        if worker_id is None:
            return await handle_api_error(request, "Missing worker_id", 400)
            
        config = load_config()
        worker_found = False
        
        for worker in config.get("workers", []):
            if worker["id"] == worker_id:
                # Update all provided fields
                if "enabled" in data:
                    worker["enabled"] = data["enabled"]
                if "name" in data:
                    worker["name"] = data["name"]
                if "port" in data:
                    worker["port"] = data["port"]
                    
                # Handle host field - remove it if None
                if "host" in data:
                    if data["host"] is None:
                        worker.pop("host", None)
                    else:
                        worker["host"] = data["host"]
                        
                # Handle cuda_device field - remove it if None
                if "cuda_device" in data:
                    if data["cuda_device"] is None:
                        worker.pop("cuda_device", None)
                    else:
                        worker["cuda_device"] = data["cuda_device"]
                        
                # Handle extra_args field - remove it if None
                if "extra_args" in data:
                    if data["extra_args"] is None:
                        worker.pop("extra_args", None)
                    else:
                        worker["extra_args"] = data["extra_args"]
                        
                # Handle type field
                if "type" in data:
                    worker["type"] = data["type"]
                        
                worker_found = True
                break
                
        if not worker_found:
            # If worker not found and all required fields are provided, create new worker
            if all(key in data for key in ["name", "port", "cuda_device"]):
                new_worker = {
                    "id": worker_id,
                    "name": data["name"],
                    "host": data.get("host", "localhost"),
                    "port": data["port"],
                    "cuda_device": data["cuda_device"],
                    "enabled": data.get("enabled", False),
                    "extra_args": data.get("extra_args", ""),
                    "type": data.get("type", "local")
                }
                if "workers" not in config:
                    config["workers"] = []
                config["workers"].append(new_worker)
                worker_found = True
            else:
                return await handle_api_error(request, f"Worker {worker_id} not found and missing required fields for creation", 404)
            
        if save_config(config):
            return web.json_response({"status": "success"})
        else:
            return await handle_api_error(request, "Failed to save config")
    except Exception as e:
        return await handle_api_error(request, e, 400)

@server.PromptServer.instance.routes.post("/distributed/config/delete_worker")
async def delete_worker_endpoint(request):
    try:
        data = await request.json()
        worker_id = data.get("worker_id")
        
        if worker_id is None:
            return await handle_api_error(request, "Missing worker_id", 400)
            
        config = load_config()
        workers = config.get("workers", [])
        
        # Find and remove the worker
        worker_index = -1
        for i, worker in enumerate(workers):
            if worker["id"] == worker_id:
                worker_index = i
                break
                
        if worker_index == -1:
            return await handle_api_error(request, f"Worker {worker_id} not found", 404)
            
        # Remove the worker
        removed_worker = workers.pop(worker_index)
        
        if save_config(config):
            return web.json_response({
                "status": "success",
                "message": f"Worker {removed_worker.get('name', worker_id)} deleted"
            })
        else:
            return await handle_api_error(request, "Failed to save config")
    except Exception as e:
        return await handle_api_error(request, e, 400)

@server.PromptServer.instance.routes.post("/distributed/config/update_setting")
async def update_setting_endpoint(request):
    """Updates a specific key in the settings object."""
    try:
        data = await request.json()
        key = data.get("key")
        value = data.get("value")

        if not key or value is None:
            return await handle_api_error(request, "Missing 'key' or 'value' in request", 400)

        config = load_config()
        if 'settings' not in config:
            config['settings'] = {}
        
        config['settings'][key] = value

        if save_config(config):
            return web.json_response({"status": "success", "message": f"Setting '{key}' updated."})
        else:
            return await handle_api_error(request, "Failed to save config")
    except Exception as e:
        return await handle_api_error(request, e, 400)

@server.PromptServer.instance.routes.post("/distributed/config/update_master")
async def update_master_endpoint(request):
    """Updates master configuration."""
    try:
        data = await request.json()
        
        config = load_config()
        if 'master' not in config:
            config['master'] = {}
        
        # Update all provided fields
        if "name" in data:
            config['master']['name'] = data['name']
        if "host" in data:
            config['master']['host'] = data['host']
        if "port" in data:
            config['master']['port'] = data['port']
        if "cuda_device" in data:
            config['master']['cuda_device'] = data['cuda_device']
        if "extra_args" in data:
            config['master']['extra_args'] = data['extra_args']
            
        if save_config(config):
            return web.json_response({"status": "success", "message": "Master configuration updated."})
        else:
            return await handle_api_error(request, "Failed to save config")
    except Exception as e:
        return await handle_api_error(request, e, 400)

@server.PromptServer.instance.routes.post("/distributed/prepare_job")
async def prepare_job_endpoint(request):
    try:
        data = await request.json()
        multi_job_id = data.get('multi_job_id')
        if not multi_job_id:
            return await handle_api_error(request, "Missing multi_job_id", 400)

        async with prompt_server.distributed_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_jobs:
                prompt_server.distributed_pending_jobs[multi_job_id] = asyncio.Queue()
        
        debug_log(f"Prepared queue for job {multi_job_id}")
        return web.json_response({"status": "success"})
    except Exception as e:
        return await handle_api_error(request, e)

@server.PromptServer.instance.routes.post("/distributed/clear_memory")
async def clear_memory_endpoint(request):
    debug_log("Received request to clear VRAM.")
    try:
        # Use ComfyUI's prompt server queue system like the /free endpoint does
        if hasattr(server.PromptServer.instance, 'prompt_queue'):
            server.PromptServer.instance.prompt_queue.set_flag("unload_models", True)
            server.PromptServer.instance.prompt_queue.set_flag("free_memory", True)
            debug_log("Set queue flags for memory clearing.")
        
        # Wait a bit for the queue to process
        await asyncio.sleep(MEMORY_CLEAR_DELAY)
        
        # Also do direct cleanup as backup, but with error handling
        import gc
        import comfy.model_management as mm
        
        try:
            mm.unload_all_models()
        except AttributeError as e:
            debug_log(f"Warning during model unload: {e}")
        
        try:
            mm.soft_empty_cache()
        except Exception as e:
            debug_log(f"Warning during cache clear: {e}")
        
        for _ in range(3):
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        debug_log("VRAM cleared successfully.")
        return web.json_response({"status": "success", "message": "GPU memory cleared."})
    except Exception as e:
        # Even if there's an error, try to do basic cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        debug_log(f"Partial VRAM clear completed with warning: {e}")
        return web.json_response({"status": "success", "message": "GPU memory cleared (with warnings)"})


@server.PromptServer.instance.routes.post("/distributed/launch_worker")
async def launch_worker_endpoint(request):
    """Launch a worker process from the UI."""
    try:
        data = await request.json()
        worker_id = data.get("worker_id")
        
        if not worker_id:
            return await handle_api_error(request, "Missing worker_id", 400)
        
        # Find worker config
        config = load_config()
        worker = next((w for w in config.get("workers", []) if w["id"] == worker_id), None)
        if not worker:
            return await handle_api_error(request, f"Worker {worker_id} not found", 404)
        
        # Ensure consistent string ID
        worker_id_str = str(worker_id)
        
        # Check if already running (managed by this instance)
        if worker_id_str in worker_manager.processes:
            proc_info = worker_manager.processes[worker_id_str]
            process = proc_info.get('process')
            
            # Check if still running
            is_running = False
            if process:
                is_running = process.poll() is None
            else:
                # Restored process without subprocess object
                is_running = worker_manager._is_process_running(proc_info['pid'])
            
            if is_running:
                return web.json_response({
                    "status": "error",
                    "message": "Worker already running (managed by UI)",
                    "pid": proc_info['pid'],
                    "log_file": proc_info.get('log_file')
                }, status=409)
            else:
                # Process is dead, remove it
                del worker_manager.processes[worker_id_str]
                worker_manager.save_processes()
        
        # Launch the worker
        try:
            pid = worker_manager.launch_worker(worker)
            log_file = worker_manager.processes[worker_id_str].get('log_file')
            return web.json_response({
                "status": "success",
                "pid": pid,
                "message": f"Worker {worker['name']} launched",
                "log_file": log_file
            })
        except Exception as e:
            return await handle_api_error(request, f"Failed to launch worker: {str(e)}", 500)
            
    except Exception as e:
        return await handle_api_error(request, e, 400)


@server.PromptServer.instance.routes.post("/distributed/stop_worker")
async def stop_worker_endpoint(request):
    """Stop a worker process that was launched from the UI."""
    try:
        data = await request.json()
        worker_id = data.get("worker_id")
        
        if not worker_id:
            return await handle_api_error(request, "Missing worker_id", 400)
        
        success, message = worker_manager.stop_worker(worker_id)
        
        if success:
            return web.json_response({"status": "success", "message": message})
        else:
            return web.json_response({"status": "error", "message": message}, 
                                   status=404 if "not managed" in message else 409)
            
    except Exception as e:
        return await handle_api_error(request, e, 400)


@server.PromptServer.instance.routes.get("/distributed/managed_workers")
async def get_managed_workers_endpoint(request):
    """Get list of workers managed by this UI instance."""
    try:
        managed = worker_manager.get_managed_workers()
        return web.json_response({
            "status": "success",
            "managed_workers": managed
        })
    except Exception as e:
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.get("/distributed/local-worker-status")
async def get_local_worker_status_endpoint(request):
    """Check status of all local workers (localhost/no host specified)."""
    try:
        config = load_config()
        worker_statuses = {}
        
        for worker in config.get("workers", []):
            # Only check local workers
            if not worker.get("host") or worker.get("host") in ["localhost", "127.0.0.1"]:
                worker_id = worker["id"]
                port = worker["port"]
                
                # Check if worker is enabled
                if not worker.get("enabled", False):
                    worker_statuses[worker_id] = {
                        "online": False,
                        "enabled": False,
                        "processing": False,
                        "queue_count": 0
                    }
                    continue
                
                # Try to connect to worker
                try:
                    session = await get_client_session()
                    async with session.get(
                        f"http://localhost:{port}/prompt",
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            queue_remaining = data.get("exec_info", {}).get("queue_remaining", 0)
                            worker_statuses[worker_id] = {
                                "online": True,
                                "enabled": True,
                                "processing": queue_remaining > 0,
                                "queue_count": queue_remaining
                            }
                        else:
                            worker_statuses[worker_id] = {
                                "online": False,
                                "enabled": True,
                                "processing": False,
                                "queue_count": 0,
                                "error": f"HTTP {resp.status}"
                            }
                except asyncio.TimeoutError:
                    worker_statuses[worker_id] = {
                        "online": False,
                        "enabled": True,
                        "processing": False,
                        "queue_count": 0,
                        "error": "Timeout"
                    }
                except Exception as e:
                    worker_statuses[worker_id] = {
                        "online": False,
                        "enabled": True,
                        "processing": False,
                        "queue_count": 0,
                        "error": str(e)
                    }
        
        return web.json_response({
            "status": "success",
            "worker_statuses": worker_statuses
        })
    except Exception as e:
        debug_log(f"Error checking local worker status: {e}")
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.get("/distributed/worker_log/{worker_id}")
async def get_worker_log_endpoint(request):
    """Get log content for a specific worker."""
    try:
        worker_id = request.match_info['worker_id']
        
        # Ensure worker_id is string
        worker_id = str(worker_id)
        
        # Check if we manage this worker
        if worker_id not in worker_manager.processes:
            return await handle_api_error(request, f"Worker {worker_id} not managed by UI", 404)
        
        proc_info = worker_manager.processes[worker_id]
        log_file = proc_info.get('log_file')
        
        if not log_file or not os.path.exists(log_file):
            return await handle_api_error(request, "Log file not found", 404)
        
        # Read last N lines (or full file if small)
        lines_to_read = int(request.query.get('lines', 1000))
        
        try:
            # Get file size
            file_size = os.path.getsize(log_file)
            
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                if lines_to_read > 0 and file_size > 1024 * 1024:  # If file > 1MB and limited lines requested
                    # Read last N lines efficiently
                    lines = []
                    # Start from end and work backwards
                    f.seek(0, 2)  # Go to end
                    file_length = f.tell()
                    
                    # Read chunks from end
                    chunk_size = min(CHUNK_SIZE, file_length)
                    while len(lines) < lines_to_read and f.tell() > 0:
                        # Move back and read chunk
                        current_pos = max(0, f.tell() - chunk_size)
                        f.seek(current_pos)
                        chunk = f.read(chunk_size)
                        
                        # Process chunk
                        chunk_lines = chunk.splitlines()
                        if current_pos > 0:
                            # Partial line at beginning, combine with next chunk
                            chunk_lines = chunk_lines[1:]
                        
                        lines = chunk_lines + lines
                        
                        # Move back for next chunk
                        f.seek(current_pos)
                    
                    # Take only last N lines
                    content = '\n'.join(lines[-lines_to_read:])
                    truncated = len(lines) > lines_to_read
                else:
                    # Read entire file
                    content = f.read()
                    truncated = False
            
            return web.json_response({
                "status": "success",
                "content": content,
                "log_file": log_file,
                "file_size": file_size,
                "truncated": truncated,
                "lines_shown": lines_to_read if truncated else content.count('\n') + 1
            })
            
        except Exception as e:
            return await handle_api_error(request, f"Error reading log file: {str(e)}", 500)
            
    except Exception as e:
        return await handle_api_error(request, e, 500)


# --- Worker Process Management ---
class WorkerProcessManager:
    def __init__(self):
        self.processes = {}  # worker_id -> process info
        self.load_processes()
        
    def find_comfy_root(self):
        """Find the ComfyUI root directory."""
        # Start from current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Method 1: Check for environment variable override
        env_root = os.environ.get('COMFYUI_ROOT')
        if env_root and os.path.exists(os.path.join(env_root, "main.py")):
            debug_log(f"Found ComfyUI root via COMFYUI_ROOT environment variable: {env_root}")
            return env_root
        
        # Method 2: Try going up from custom_nodes directory
        # This file should be in ComfyUI/custom_nodes/ComfyUI-Distributed/
        potential_root = os.path.dirname(os.path.dirname(current_dir))
        if os.path.exists(os.path.join(potential_root, "main.py")):
            debug_log(f"Found ComfyUI root via directory traversal: {potential_root}")
            return potential_root
        
        # Method 3: Look for common Docker paths
        docker_paths = ["/basedir", "/ComfyUI", "/app", "/workspace/ComfyUI", "/comfyui", "/opt/ComfyUI", "/workspace"]
        for path in docker_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "main.py")):
                debug_log(f"Found ComfyUI root in Docker path: {path}")
                return path
        
        # Method 4: Search upwards for main.py
        search_dir = current_dir
        for _ in range(5):  # Limit search depth
            if os.path.exists(os.path.join(search_dir, "main.py")):
                debug_log(f"Found ComfyUI root via upward search: {search_dir}")
                return search_dir
            parent = os.path.dirname(search_dir)
            if parent == search_dir:  # Reached root
                break
            search_dir = parent
        
        # Method 5: Try to import and use folder_paths
        try:
            import folder_paths
            # folder_paths.base_path should point to ComfyUI root
            if hasattr(folder_paths, 'base_path') and os.path.exists(os.path.join(folder_paths.base_path, "main.py")):
                debug_log(f"Found ComfyUI root via folder_paths: {folder_paths.base_path}")
                return folder_paths.base_path
        except:
            pass
        
        # If all methods fail, log detailed information and return the best guess
        log(f"Warning: Could not reliably determine ComfyUI root directory")
        log(f"Current directory: {current_dir}")
        log(f"Initial guess was: {potential_root}")
        
        # Return the initial guess as fallback
        return potential_root
        
    def _find_windows_terminal(self):
        """Find Windows Terminal executable."""
        # Common locations for Windows Terminal
        possible_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\wt.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\WindowsApps\Microsoft.WindowsTerminal_*\wt.exe"),
            "wt.exe"  # Try PATH
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
            # Handle wildcard for WindowsApps
            if '*' in path:
                import glob
                matches = glob.glob(path)
                if matches:
                    return matches[0]
        
        # Try to find it in PATH
        import shutil
        wt_path = shutil.which("wt")
        if wt_path:
            return wt_path
            
        return None
        
    def build_launch_command(self, worker_config, comfy_root):
        """Build the command to launch a worker."""
        # Use main.py directly - it's the most reliable method
        main_py = os.path.join(comfy_root, "main.py")
        
        if os.path.exists(main_py):
            cmd = [
                get_python_executable(),
                main_py,
                "--port", str(worker_config['port']),
                "--enable-cors-header"
            ]
            debug_log(f"Using main.py: {main_py}")
        else:
            # Provide detailed error message
            error_msg = f"Could not find main.py in {comfy_root}\n"
            error_msg += f"Searched for: {main_py}\n"
            error_msg += f"Directory contents of {comfy_root}:\n"
            try:
                if os.path.exists(comfy_root):
                    files = os.listdir(comfy_root)[:20]  # List first 20 files
                    error_msg += "  " + "\n  ".join(files)
                    if len(os.listdir(comfy_root)) > 20:
                        error_msg += f"\n  ... and {len(os.listdir(comfy_root)) - 20} more files"
                else:
                    error_msg += f"  Directory {comfy_root} does not exist!"
            except Exception as e:
                error_msg += f"  Error listing directory: {e}"
            
            # Try to suggest the correct path
            error_msg += "\n\nPossible solutions:\n"
            error_msg += "1. Check if ComfyUI is installed in a different location\n"
            error_msg += "2. For Docker: ComfyUI might be in /ComfyUI or /app\n"
            error_msg += "3. Ensure the custom node is installed in the correct location\n"
            
            raise RuntimeError(error_msg)
        
        # Add any extra arguments safely
        if worker_config.get('extra_args'):
            raw_args = worker_config['extra_args'].strip()
            if raw_args:
                # Safely split using shlex to handle quotes and spaces
                extra_args_list = shlex.split(raw_args)
                
                # Validate: Block dangerous shell meta-characters (e.g., ; | & > < ` $)
                forbidden_chars = set(';|>&<`$()[]{}*!?')
                for arg in extra_args_list:
                    if any(c in forbidden_chars for c in arg):
                        raise ValueError(f"Invalid characters in extra_args: {arg}. Forbidden: {''.join(forbidden_chars)}")
                
                cmd.extend(extra_args_list)
            
        return cmd
        
    def launch_worker(self, worker_config, show_window=False):
        """Launch a worker process with logging."""
        comfy_root = self.find_comfy_root()
        
        # Set up environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(worker_config.get('cuda_device', 0))
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Pass master PID to worker so it can monitor if master is still alive
        env['COMFYUI_MASTER_PID'] = str(os.getpid())
        
        cmd = self.build_launch_command(worker_config, comfy_root)
        
        # Change to ComfyUI root directory for the process
        cwd = comfy_root
        
        # Create log directory and file
        log_dir = os.path.join(comfy_root, "logs", "workers")
        os.makedirs(log_dir, exist_ok=True)
        
        # Use daily log files instead of timestamp
        date_stamp = time.strftime("%Y%m%d")
        worker_name = worker_config.get('name', f'Worker{worker_config["id"]}')
        # Clean worker name for filename
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in worker_name)
        log_file = os.path.join(log_dir, f"{safe_name}_{date_stamp}.log")
        
        # Launch process with logging (append mode for daily logs)
        with open(log_file, 'a') as log_handle:
            # Write startup info to log with timestamp
            log_handle.write(f"\n\n{'='*50}\n")
            log_handle.write(f"=== ComfyUI Worker Session Started ===\n")
            log_handle.write(f"Worker: {worker_name}\n")
            log_handle.write(f"Port: {worker_config['port']}\n")
            log_handle.write(f"CUDA Device: {worker_config.get('cuda_device', 0)}\n")
            log_handle.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_handle.write(f"Command: {' '.join(cmd)}\n")
            
            # Note about worker behavior
            config = load_config()
            stop_on_master_exit = config.get('settings', {}).get('stop_workers_on_master_exit', True)
            
            if stop_on_master_exit:
                log_handle.write("Note: Worker will stop when master shuts down\n")
            else:
                log_handle.write("Note: Worker will continue running after master shuts down\n")
            
            log_handle.write("=" * 30 + "\n\n")
            log_handle.flush()
            
            # Wrap command with monitor if needed
            if stop_on_master_exit and env.get('COMFYUI_MASTER_PID'):
                # Use the monitor wrapper
                monitor_script = os.path.join(os.path.dirname(__file__), 'worker_monitor.py')
                monitored_cmd = [get_python_executable(), monitor_script] + cmd
                log_handle.write(f"[Worker Monitor] Monitoring master PID: {env['COMFYUI_MASTER_PID']}\n")
                log_handle.flush()
            else:
                monitored_cmd = cmd
            
            # Platform-specific process creation - always hidden with logging
            if platform.system() == "Windows":
                CREATE_NO_WINDOW = 0x08000000
                process = subprocess.Popen(
                    monitored_cmd, env=env, cwd=cwd,
                    stdout=log_handle, 
                    stderr=subprocess.STDOUT,
                    creationflags=CREATE_NO_WINDOW
                )
            else:
                # Unix-like systems
                process = subprocess.Popen(
                    monitored_cmd, env=env, cwd=cwd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True  # Detach from parent
                )
        
        # Track the process with log file info - use string ID for consistency
        worker_id = str(worker_config['id'])
        self.processes[worker_id] = {
            'pid': process.pid,
            'process': process,
            'started_at': time.time(),
            'config': worker_config,
            'log_file': log_file,
            'is_monitor': stop_on_master_exit and env.get('COMFYUI_MASTER_PID'),  # Track if using monitor
            'launching': True  # Mark as launching until confirmed running
        }
        
        # Save process info for persistence
        self.save_processes()
        
        if stop_on_master_exit and env.get('COMFYUI_MASTER_PID'):
            debug_log(f"Launched worker {worker_name} via monitor (Monitor PID: {process.pid})")
        else:
            log(f"Launched worker {worker_name} directly (PID: {process.pid})")
        debug_log(f"Log file: {log_file}")
        return process.pid
        
    def stop_worker(self, worker_id):
        """Stop a worker process."""
        # Ensure worker_id is string
        worker_id = str(worker_id)
        if worker_id not in self.processes:
            return False, "Worker not managed by UI"
            
        proc_info = self.processes[worker_id]
        process = proc_info.get('process')
        pid = proc_info['pid']
        
        debug_log(f"Attempting to stop worker {worker_id} (PID: {pid})")
        
        # For restored processes without subprocess object
        if not process:
            try:
                print(f"[Distributed] Stopping restored process (no subprocess object)")
                if self._kill_process_tree(pid):
                    del self.processes[worker_id]
                    self.save_processes()
                    debug_log(f"Successfully stopped worker {worker_id} and all child processes")
                    return True, "Worker stopped"
                else:
                    return False, "Failed to stop worker process"
            except Exception as e:
                print(f"[MultiGPU] Exception during stop: {e}")
                return False, f"Error stopping worker: {str(e)}"
        
        # Normal case with subprocess object
        # Check if still running
        if process.poll() is not None:
            # Already stopped
            print(f"[Distributed] Worker {worker_id} already stopped")
            del self.processes[worker_id]
            self.save_processes()
            return False, "Worker already stopped"
            
        # Try to kill the entire process tree
        try:
            debug_log(f"Using process tree kill for worker {worker_id}")
            if self._kill_process_tree(pid):
                # Clean up tracking
                del self.processes[worker_id]
                self.save_processes()
                debug_log(f"Successfully stopped worker {worker_id} and all child processes")
                return True, "Worker stopped"
            else:
                # Fallback to normal termination
                print(f"[Distributed] Process tree kill failed, trying normal termination")
                if process:
                    terminate_process(process, timeout=PROCESS_TERMINATION_TIMEOUT)
                
                del self.processes[worker_id]
                self.save_processes()
                return True, "Worker stopped (fallback)"
                
        except Exception as e:
            print(f"[Distributed] Exception during stop: {e}")
            return False, f"Error stopping worker: {str(e)}"
            
    def get_managed_workers(self):
        """Get list of workers managed by this process."""
        managed = {}
        for worker_id, proc_info in list(self.processes.items()):
            # Check if process is still running
            is_running, _ = self._check_worker_process(worker_id, proc_info)
            
            if is_running:
                managed[worker_id] = {
                    'pid': proc_info['pid'],
                    'started_at': proc_info['started_at'],
                    'log_file': proc_info.get('log_file'),
                    'launching': proc_info.get('launching', False)
                }
            else:
                # Process has stopped, remove from tracking
                del self.processes[worker_id]
        
        return managed
        
    def cleanup_all(self):
        """Stop all managed workers (called on shutdown)."""
        for worker_id in list(self.processes.keys()):
            try:
                self.stop_worker(worker_id)
            except Exception as e:
                print(f"[Distributed] Error stopping worker {worker_id}: {e}")
        
        # Clear all managed processes from config
        config = load_config()
        config['managed_processes'] = {}
        save_config(config)
    
    def load_processes(self):
        """Load persisted process information from config."""
        config = load_config()
        managed_processes = config.get('managed_processes', {})
        
        # Verify each saved process is still running
        for worker_id, proc_info in managed_processes.items():
            pid = proc_info.get('pid')
            if pid and self._is_process_running(pid):
                # Reconstruct process info
                self.processes[worker_id] = {
                    'pid': pid,
                    'process': None,  # Can't reconstruct subprocess object
                    'started_at': proc_info.get('started_at'),
                    'config': proc_info.get('config'),
                    'log_file': proc_info.get('log_file')
                }
                print(f"[Distributed] Restored worker {worker_id} (PID: {pid})")
            else:
                if pid:
                    print(f"[Distributed] Worker {worker_id} (PID: {pid}) is no longer running")
    
    def save_processes(self):
        """Save process information to config."""
        config = load_config()
        
        # Create serializable version of process info
        managed_processes = {}
        for worker_id, proc_info in self.processes.items():
            # Only save if process is running
            is_running, _ = self._check_worker_process(worker_id, proc_info)
            
            if is_running:
                managed_processes[worker_id] = {
                    'pid': proc_info['pid'],
                    'started_at': proc_info['started_at'],
                    'config': proc_info['config'],
                    'log_file': proc_info.get('log_file'),
                    'launching': proc_info.get('launching', False)
                }
        
        # Update config with managed processes
        config['managed_processes'] = managed_processes
        save_config(config)
    
    def _is_process_running(self, pid):
        """Check if a process with given PID is running."""
        return is_process_alive(pid)
    
    def _check_worker_process(self, worker_id, proc_info):
        """Check if a worker process is still running and return status.
        
        Returns:
            tuple: (is_running, has_subprocess_object)
        """
        process = proc_info.get('process')
        pid = proc_info.get('pid')
        
        if process:
            # Normal case with subprocess object
            return process.poll() is None, True
        elif pid:
            # Restored process without subprocess object
            return self._is_process_running(pid), False
        else:
            # No process or PID
            return False, False
    
    def _kill_process_tree(self, pid):
        """Kill a process and all its children."""
        if PSUTIL_AVAILABLE:
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                
                # Log what we're about to kill
                debug_log(f"Killing process tree for PID {pid} ({parent.name()})")
                for child in children:
                    debug_log(f"  - Child PID {child.pid} ({child.name()})")
                
                # Kill children first
                for child in children:
                    try:
                        debug_log(f"Terminating child {child.pid}")
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                
                # Wait a bit for graceful termination
                gone, alive = psutil.wait_procs(children, timeout=PROCESS_WAIT_TIMEOUT)
                
                # Force kill any remaining
                for child in alive:
                    try:
                        debug_log(f"Force killing child {child.pid}")
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                
                # Finally kill the parent
                try:
                    debug_log(f"Terminating parent {pid}")
                    parent.terminate()
                    parent.wait(timeout=PROCESS_WAIT_TIMEOUT)
                except psutil.TimeoutExpired:
                    debug_log(f"Force killing parent {pid}")
                    parent.kill()
                except psutil.NoSuchProcess:
                    debug_log(f"Parent process {pid} already gone")
                    
                return True
                
            except psutil.NoSuchProcess:
                debug_log(f"Process {pid} does not exist")
                return False
            except Exception as e:
                debug_log(f"Error killing process tree: {e}")
                # Fall through to OS commands
        
        # Fallback to OS-specific commands
        print(f"[Distributed] Using OS commands to kill process tree")
        if platform.system() == "Windows":
            try:
                # Use wmic to find child processes
                result = subprocess.run(['wmic', 'process', 'where', f'ParentProcessId={pid}', 'get', 'ProcessId'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    child_pids = [line.strip() for line in lines if line.strip() and line.strip().isdigit()]
                    
                    print(f"[Distributed] Found child processes: {child_pids}")
                    
                    # Kill each child
                    for child_pid in child_pids:
                        try:
                            subprocess.run(['taskkill', '/F', '/PID', child_pid], 
                                         capture_output=True, check=False)
                        except:
                            pass
                
                # Kill the parent with tree flag
                result = subprocess.run(['taskkill', '/F', '/PID', str(pid), '/T'], 
                                      capture_output=True, text=True)
                print(f"[Distributed] Taskkill result: {result.stdout.strip()}")
                return result.returncode == 0
            except Exception as e:
                print(f"[Distributed] Error with taskkill: {e}")
                return False
        else:
            # Unix: use pkill
            try:
                subprocess.run(['pkill', '-TERM', '-P', str(pid)], check=False)
                time.sleep(WORKER_CHECK_INTERVAL)
                subprocess.run(['pkill', '-KILL', '-P', str(pid)], check=False)
                os.kill(pid, signal.SIGKILL)
                return True
            except:
                return False

# Create global instance
worker_manager = WorkerProcessManager()

# Add IPC queue storage
worker_manager.queues = {}

# Helper to get aiohttp session
async def get_client_session():
    """Get or create aiohttp client session."""
    if not hasattr(server.PromptServer.instance, '_distributed_session'):
        server.PromptServer.instance._distributed_session = aiohttp.ClientSession()
    return server.PromptServer.instance._distributed_session

# Local worker detection functions
async def is_local_worker(worker_config):
    """Check if a worker is running on the same machine as the master."""
    host = worker_config.get('host', 'localhost')
    if host in ['localhost', '127.0.0.1', '0.0.0.0', ''] or worker_config.get('type') == 'local':
        return True
    
    # For cloud workers, check if on same physical host
    if worker_config.get('type') == 'cloud':
        return await is_same_physical_host(worker_config)
    
    return False

async def is_same_physical_host(worker_config):
    """Compare machine IDs to determine if worker is on same physical host."""
    try:
        # Get master machine ID
        master_machine_id = get_machine_id()
        
        # Fetch worker's machine ID via API
        host = worker_config.get('host', 'localhost')
        port = worker_config.get('port', 8188)
        
        session = await get_client_session()
        async with session.get(
            f"http://{host}:{port}/distributed/system_info",
            timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                worker_machine_id = data.get('machine_id')
                return worker_machine_id == master_machine_id
            else:
                debug_log(f"Failed to get system info from worker: HTTP {resp.status}")
                return False
    except Exception as e:
        debug_log(f"Error checking same physical host: {e}")
        return False

def get_machine_id():
    """Get a unique identifier for this machine."""
    # Try multiple methods to get a stable machine ID
    try:
        # Method 1: MAC address-based UUID
        return str(uuid.getnode())
    except:
        try:
            # Method 2: Platform + hostname
            import socket
            return f"{platform.machine()}_{socket.gethostname()}"
        except:
            # Fallback
            return platform.machine()

def is_docker_environment():
    """Check if running inside Docker container."""
    return (os.path.exists('/.dockerenv') or 
            os.environ.get('DOCKER_CONTAINER', False) or
            'docker' in platform.node().lower())

def is_runpod_environment():
    """Check if running in Runpod environment."""
    return (os.environ.get('RUNPOD_POD_ID') is not None or
            os.environ.get('RUNPOD_API_KEY') is not None)

async def get_comms_channel(worker_id, worker_config):
    """Get communication channel for a worker (HTTP URL or IPC Queue)."""
    if await is_local_worker(worker_config):
        comm_mode = worker_config.get('communicationMode', 'http')
        
        if comm_mode == 'ipc':
            # Use multiprocessing Queue for IPC
            if worker_id not in worker_manager.queues:
                worker_manager.queues[worker_id] = Queue()
            return worker_manager.queues[worker_id]
        elif is_docker_environment():
            # Docker: use host.docker.internal
            return f"http://host.docker.internal:{worker_config['port']}"
        else:
            # Local: use loopback
            return f"http://127.0.0.1:{worker_config['port']}"
    elif worker_config.get('type') == 'cloud' and is_runpod_environment():
        # Runpod same-host optimization (if detected)
        host = worker_config.get('host', 'localhost')
        return f"http://{host}:{worker_config['port']}"
    else:
        # Remote worker: use configured endpoint
        host = worker_config.get('host', 'localhost')
        return f"http://{host}:{worker_config['port']}"

# Auto-launch workers if enabled
def auto_launch_workers():
    """Launch enabled workers if auto_launch_workers is set to true."""
    try:
        config = load_config()
        if config.get('settings', {}).get('auto_launch_workers', False):
            log("Auto-launch workers is enabled, checking for workers to start...")
            
            # Clear managed_processes before launching new workers
            # This handles cases where the master was killed without proper cleanup
            if config.get('managed_processes'):
                log("Clearing old managed_processes before auto-launch...")
                config['managed_processes'] = {}
                save_config(config)
            
            workers = config.get('workers', [])
            launched_count = 0
            
            for worker in workers:
                if worker.get('enabled', False):
                    worker_id = worker.get('id')
                    worker_name = worker.get('name', f'Worker {worker_id}')
                    
                    # Skip remote workers
                    host = worker.get('host', 'localhost').lower()
                    if host not in ['localhost', '127.0.0.1', '', None]:
                        debug_log(f"Skipping remote worker {worker_name} (host: {host})")
                        continue
                    
                    # Check if already running
                    if str(worker_id) in worker_manager.processes:
                        proc_info = worker_manager.processes[str(worker_id)]
                        if worker_manager._is_process_running(proc_info['pid']):
                            debug_log(f"Worker {worker_name} already running, skipping")
                            continue
                    
                    # Launch the worker
                    try:
                        pid = worker_manager.launch_worker(worker)
                        log(f"Auto-launched worker {worker_name} (PID: {pid})")
                        
                        # Mark as launching in managed processes
                        if str(worker_id) in worker_manager.processes:
                            worker_manager.processes[str(worker_id)]['launching'] = True
                            worker_manager.save_processes()
                        
                        launched_count += 1
                    except Exception as e:
                        log(f"Failed to auto-launch worker {worker_name}: {e}")
            
            if launched_count > 0:
                log(f"Auto-launched {launched_count} worker(s)")
            else:
                debug_log("No workers to auto-launch")
        else:
            debug_log("Auto-launch workers is disabled")
    except Exception as e:
        log(f"Error during auto-launch: {e}")

# Schedule auto-launch after a short delay to ensure server is ready
def delayed_auto_launch():
    """Delay auto-launch to ensure server is fully initialized."""
    import threading
    timer = threading.Timer(WORKER_STARTUP_DELAY, auto_launch_workers)
    timer.daemon = True
    timer.start()

# Async cleanup function for proper shutdown
async def async_cleanup_and_exit(signum=None):
    """Async-friendly cleanup and exit."""
    try:
        config = load_config()
        if config.get('settings', {}).get('stop_workers_on_master_exit', True):
            print("\n[Distributed] Master shutting down, stopping all managed workers...")
            worker_manager.cleanup_all()
        else:
            print("\n[Distributed] Master shutting down, workers will continue running")
            worker_manager.save_processes()
    except Exception as e:
        print(f"[Distributed] Error during cleanup: {e}")
    
    # On Windows, we need to exit differently
    if platform.system() == "Windows":
        # Force exit on Windows
        sys.exit(0)
    else:
        # On Unix, stop the event loop gracefully
        loop = asyncio.get_running_loop()
        loop.stop()

def register_async_signals():
    """Register async signal handlers for graceful shutdown."""
    if platform.system() == "Windows":
        # Windows doesn't support add_signal_handler, use traditional signal handling
        def signal_handler(signum, frame):
            # Schedule the async cleanup in the event loop
            loop = server.PromptServer.instance.loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(async_cleanup_and_exit(signum), loop)
            else:
                # Fallback to sync cleanup if loop isn't running
                try:
                    config = load_config()
                    if config.get('settings', {}).get('stop_workers_on_master_exit', True):
                        print("\n[Distributed] Master shutting down, stopping all managed workers...")
                        worker_manager.cleanup_all()
                    else:
                        print("\n[Distributed] Master shutting down, workers will continue running")
                        worker_manager.save_processes()
                except Exception as e:
                    print(f"[Distributed] Error during cleanup: {e}")
                sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        # Unix-like systems support add_signal_handler
        loop = server.PromptServer.instance.loop
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(async_cleanup_and_exit(s)))
        
        # SIGHUP is Unix-only
        loop.add_signal_handler(signal.SIGHUP, lambda: asyncio.create_task(async_cleanup_and_exit(signal.SIGHUP)))

def sync_cleanup():
    """Synchronous wrapper for atexit."""
    try:
        # For atexit, we don't want to stop the loop or exit
        config = load_config()
        if config.get('settings', {}).get('stop_workers_on_master_exit', True):
            print("\n[Distributed] Master shutting down, stopping all managed workers...")
            worker_manager.cleanup_all()
        else:
            print("\n[Distributed] Master shutting down, workers will continue running")
            worker_manager.save_processes()
    except Exception as e:
        print(f"[Distributed] Error during cleanup: {e}")

# Register atexit handler for normal exits
atexit.register(sync_cleanup)

# Call delayed auto-launch only if we're the master (not a worker)
if not os.environ.get('COMFYUI_MASTER_PID'):
    delayed_auto_launch()
    try:
        register_async_signals()  # Register async signal handlers
    except Exception as e:
        print(f"[Distributed] Warning: Could not register async signal handlers: {e}")
        # Fall back to basic signal handling
        def basic_signal_handler(signum, frame):
            print("\n[Distributed] Received signal, shutting down...")
            try:
                config = load_config()
                if config.get('settings', {}).get('stop_workers_on_master_exit', True):
                    worker_manager.cleanup_all()
                else:
                    worker_manager.save_processes()
            except Exception as cleanup_error:
                print(f"[Distributed] Error during cleanup: {cleanup_error}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, basic_signal_handler)
        signal.signal(signal.SIGTERM, basic_signal_handler)
else:
    debug_log("Running as worker, skipping auto-launch")

# --- Persistent State Storage ---
# Store job queue on the persistent server instance to survive script reloads
prompt_server = server.PromptServer.instance

# Initialize persistent state if not already present
if not hasattr(prompt_server, 'distributed_pending_jobs'):
    debug_log("Initializing persistent job queue on server instance.")
    prompt_server.distributed_pending_jobs = {}
    prompt_server.distributed_jobs_lock = asyncio.Lock()

@server.PromptServer.instance.routes.post("/distributed/load_image")
async def load_image_endpoint(request):
    """Load an image or video file and return it as base64 data with hash."""
    try:
        data = await request.json()
        image_path = data.get("image_path")
        
        if not image_path:
            return await handle_api_error(request, "Missing image_path", 400)
        
        import folder_paths
        import base64
        import io
        import hashlib
        
        # Use ComfyUI's folder paths to find the file
        full_path = folder_paths.get_annotated_filepath(image_path)
        
        if not os.path.exists(full_path):
            return await handle_api_error(request, f"File not found: {image_path}", 404)
        
        # Calculate file hash
        hash_md5 = hashlib.md5()
        with open(full_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        file_hash = hash_md5.hexdigest()
        
        # Check if it's a video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        file_ext = os.path.splitext(full_path)[1].lower()
        
        if file_ext in video_extensions:
            # For video files, read the raw bytes
            with open(full_path, 'rb') as f:
                file_data = f.read()
            
            # Determine MIME type
            mime_types = {
                '.mp4': 'video/mp4',
                '.avi': 'video/x-msvideo', 
                '.mov': 'video/quicktime',
                '.mkv': 'video/x-matroska',
                '.webm': 'video/webm'
            }
            mime_type = mime_types.get(file_ext, 'video/mp4')
            
            # Return base64 encoded video with data URL
            video_base64 = base64.b64encode(file_data).decode('utf-8')
            return web.json_response({
                "status": "success",
                "image_data": f"data:{mime_type};base64,{video_base64}",
                "hash": file_hash
            })
        else:
            # For images, use PIL
            from PIL import Image
            
            # Load and convert to base64
            with Image.open(full_path) as img:
                # Convert to RGB if needed
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='PNG', compress_level=1)  # Fast compression
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return web.json_response({
                "status": "success",
                "image_data": f"data:image/png;base64,{img_base64}",
                "hash": file_hash
            })
        
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.post("/distributed/check_file")
async def check_file_endpoint(request):
    """Check if a file exists and matches the given hash."""
    try:
        data = await request.json()
        filename = data.get("filename")
        expected_hash = data.get("hash")
        
        if not filename or not expected_hash:
            return await handle_api_error(request, "Missing filename or hash", 400)
        
        import folder_paths
        import hashlib
        
        # Use ComfyUI's folder paths to find the file
        full_path = folder_paths.get_annotated_filepath(filename)
        
        if not os.path.exists(full_path):
            return web.json_response({
                "status": "success",
                "exists": False
            })
        
        # Calculate file hash
        hash_md5 = hashlib.md5()
        with open(full_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        file_hash = hash_md5.hexdigest()
        
        # Check if hash matches
        hash_matches = file_hash == expected_hash
        
        return web.json_response({
            "status": "success",
            "exists": True,
            "hash_matches": hash_matches
        })
        
    except Exception as e:
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.post("/distributed/job_complete")
async def job_complete_endpoint(request):
    try:
        data = await request.post()
        multi_job_id = data.get('multi_job_id')
        worker_id = data.get('worker_id')
        is_last = data.get('is_last', 'False').lower() == 'true'
        
        if multi_job_id is None or worker_id is None:
            return await handle_api_error(request, "Missing multi_job_id or worker_id", 400)

        # Check for batch mode
        batch_size = int(data.get('batch_size', 0))
        tensors = []
        
        if batch_size > 0:
            # Batch mode: Extract multiple images
            debug_log(f"job_complete received batch - job_id: {multi_job_id}, worker: {worker_id}, batch_size: {batch_size}")
            
            # Check for JSON metadata
            metadata_field = data.get('images_metadata')
            metadata = None
            if metadata_field:
                # New JSON metadata format
                try:
                    # Handle different types of metadata field
                    if hasattr(metadata_field, 'file'):
                        # File-like object
                        metadata_str = metadata_field.file.read().decode('utf-8')
                    elif isinstance(metadata_field, (bytes, bytearray)):
                        # Direct bytes/bytearray
                        metadata_str = metadata_field.decode('utf-8')
                    else:
                        # String
                        metadata_str = str(metadata_field)
                    
                    metadata = json.loads(metadata_str)
                    if len(metadata) != batch_size:
                        return await handle_api_error(request, "Metadata length mismatch", 400)
                    debug_log(f"Using JSON metadata for batch from worker {worker_id}")
                except Exception as e:
                    log(f"Error parsing JSON metadata from worker {worker_id}: {e}")
                    return await handle_api_error(request, f"Metadata parsing error: {e}", 400)
            else:
                # Legacy format - log deprecation warning
                debug_log(f"WARNING: Worker {worker_id} using legacy field format. Please update to use JSON metadata.")
            
            # Process images with per-item error handling
            image_data_list = []  # List of tuples: (index, tensor)
            
            for i in range(batch_size):
                image_field = data.get(f'image_{i}')
                if image_field is None:
                    log(f"Missing image_{i} from worker {worker_id}, skipping")
                    continue
                    
                try:
                    img_data = image_field.file.read()
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    tensor = pil_to_tensor(img)
                    tensor = ensure_contiguous(tensor)
                    
                    # Get expected index from metadata or use position
                    if metadata and i < len(metadata):
                        expected_idx = metadata[i].get('index', i)
                        if i != expected_idx:
                            debug_log(f"Warning: Image order mismatch at position {i}, expected index {expected_idx}")
                    else:
                        expected_idx = i
                    
                    image_data_list.append((expected_idx, tensor))
                except Exception as e:
                    log(f"Error processing image {i} from worker {worker_id}: {e}, skipping this image")
                    # Continue processing other images instead of failing entire batch
                    continue
            
            # Check if we got any valid images
            if not image_data_list:
                return await handle_api_error(request, "No valid images in batch", 400)
            
            # Sort by index to ensure correct order
            image_data_list.sort(key=lambda x: x[0])
            tensors = [tensor for _, tensor in image_data_list]
            
            # Keep the indices for later use
            indices = [idx for idx, _ in image_data_list]
            
            # Validate final order
            if metadata:
                debug_log(f"Reordered {len(tensors)} images based on metadata indices")
        else:
            # Fallback for single-image mode (backward compatibility)
            image_file = data.get('image')
            image_index = data.get('image_index')
            
            debug_log(f"job_complete received single - job_id: {multi_job_id}, worker: {worker_id}, index: {image_index}, is_last: {is_last}")
            
            if not image_file:
                return await handle_api_error(request, "Missing image data", 400)
                
            try:
                img_data = image_file.file.read()
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                tensor = pil_to_tensor(img)
                tensor = ensure_contiguous(tensor)
                tensors = [tensor]
            except Exception as e:
                log(f"Error processing image from worker {worker_id}: {e}")
                return await handle_api_error(request, f"Image processing error: {e}", 400)

        # Put batch into queue
        async with prompt_server.distributed_jobs_lock:
            debug_log(f"Current pending jobs: {list(prompt_server.distributed_pending_jobs.keys())}")
            if multi_job_id in prompt_server.distributed_pending_jobs:
                if batch_size > 0:
                    # Put batch as single item with indices
                    await prompt_server.distributed_pending_jobs[multi_job_id].put({
                        'worker_id': worker_id,
                        'tensors': tensors,
                        'indices': indices,
                        'is_last': is_last
                    })
                    debug_log(f"Received batch result for job {multi_job_id} from worker {worker_id}, size={len(tensors)}")
                else:
                    # Put single image (backward compat)
                    await prompt_server.distributed_pending_jobs[multi_job_id].put({
                        'tensor': tensors[0],
                        'worker_id': worker_id,
                        'image_index': int(image_index) if image_index else 0,
                        'is_last': is_last
                    })
                    debug_log(f"Received single result for job {multi_job_id} from worker {worker_id}")
                    
                debug_log(f"Queue size after put: {prompt_server.distributed_pending_jobs[multi_job_id].qsize()}")
                return web.json_response({"status": "success"})
            else:
                log(f"ERROR: Job {multi_job_id} not found in distributed_pending_jobs")
                return await handle_api_error(request, "Job not found or already complete", 404)
    except Exception as e:
        return await handle_api_error(request, e)


# --- Collector Node ---
class DistributedCollectorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "images": ("IMAGE",) },
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "worker_batch_size": ("INT", {"default": 1, "min": 1, "max": 1024}),
                "worker_id": ("STRING", {"default": ""}),
                "pass_through": ("BOOLEAN", {"default": False}),
                "delegate_only": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image"
    
    def run(self, images, multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]", worker_batch_size=1, worker_id="", pass_through=False, delegate_only=False):
        if not multi_job_id or pass_through:
            if pass_through:
                print(f"[Distributed Collector] Pass-through mode enabled, returning images unchanged")
            return (images,)

        # Use async helper to run in server loop
        result = run_async_in_server_loop(
            self.execute(images, multi_job_id, is_worker, master_url, enabled_worker_ids, worker_batch_size, worker_id, delegate_only)
        )
        return result

    async def send_batch_to_master(self, image_batch, multi_job_id, master_url, worker_id):
        """Send image batch to master, chunked if large."""
        batch_size = image_batch.shape[0]
        if batch_size == 0:
            return
        
        
        for start in range(0, batch_size, MAX_BATCH):
            chunk = image_batch[start:start + MAX_BATCH]
            chunk_size = chunk.shape[0]
            is_chunk_last = (start + chunk_size == batch_size)  # True only for final chunk
            
            
            data = aiohttp.FormData()
            data.add_field('multi_job_id', multi_job_id)
            data.add_field('worker_id', str(worker_id))
            data.add_field('is_last', str(is_chunk_last))
            data.add_field('batch_size', str(chunk_size))
            
            # Chunk metadata: Absolute index from full batch
            metadata = [{'index': start + j} for j in range(chunk_size)]
            data.add_field('images_metadata', json.dumps(metadata), content_type='application/json')
            
            # Add chunk images
            for j in range(chunk_size):
                # Convert tensor slice to PIL
                img = tensor_to_pil(chunk[j:j+1], 0)
                byte_io = io.BytesIO()
                img.save(byte_io, format='PNG', compress_level=0)
                byte_io.seek(0)
                data.add_field(f'image_{j}', byte_io, filename=f'image_{j}.png', content_type='image/png')
            
            try:
                
                session = await get_client_session()
                url = f"{master_url}/distributed/job_complete"
                async with session.post(url, data=data) as response:
                    response.raise_for_status()
            except Exception as e:
                log(f"Worker - Failed to send chunk to master: {e}")
                debug_log(f"Worker - Full error details: URL={url}")
                raise  # Re-raise to handle at caller level


    async def execute(self, images, multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]", worker_batch_size=1, worker_id="", delegate_only=False):
        if is_worker:
            # Worker mode: send images to master in a single batch
            debug_log(f"Worker - Job {multi_job_id} complete. Sending {images.shape[0]} image(s) to master")
            await self.send_batch_to_master(images, multi_job_id, master_url, worker_id)
            return (images,)
        else:
            delegate_mode = delegate_only or is_master_delegate_only()
            # Master mode: collect images from workers
            enabled_workers = json.loads(enabled_worker_ids)
            num_workers = len(enabled_workers)
            if num_workers == 0:
                return (images,)
            
            if delegate_mode:
                master_batch_size = 0
                images_on_cpu = None
                debug_log(f"Master - Job {multi_job_id}: Delegate-only mode enabled, collecting exclusively from {num_workers} workers")
            else:
                images_on_cpu = images.cpu()
                master_batch_size = images.shape[0]
                debug_log(f"Master - Job {multi_job_id}: Master has {master_batch_size} images, collecting from {num_workers} workers...")
                
                # Ensure master images are contiguous
                images_on_cpu = ensure_contiguous(images_on_cpu)
            
            
            # Initialize storage for collected images
            worker_images = {}  # Dict to store images by worker_id and index
            
            # Get the existing queue - it should already exist from prepare_job
            async with prompt_server.distributed_jobs_lock:
                if multi_job_id not in prompt_server.distributed_pending_jobs:
                    log(f"Master - WARNING: Queue doesn't exist for job {multi_job_id}, creating one")
                    prompt_server.distributed_pending_jobs[multi_job_id] = asyncio.Queue()
                else:
                    existing_size = prompt_server.distributed_pending_jobs[multi_job_id].qsize()
                    debug_log(f"Master - Using existing queue for job {multi_job_id} (current size: {existing_size})")
            
            # Collect images until all workers report they're done
            collected_count = 0
            workers_done = set()
            
            # Use unified worker timeout from config/UI with simple sliced waits
            base_timeout = float(get_worker_timeout_seconds())
            slice_timeout = min(0.5, base_timeout)  # small per-wait slice to recheck interrupt
            last_activity = time.time()
            
            
            # Get queue size before starting
            async with prompt_server.distributed_jobs_lock:
                q = prompt_server.distributed_pending_jobs[multi_job_id]
                initial_size = q.qsize()

            # NEW: Initialize progress bar for workers (total = num_workers)
            p = ProgressBar(num_workers)

            try:
                while len(workers_done) < num_workers:
                    # Check for user interruption to abort collection promptly
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    try:
                        # Get the queue again each time to ensure we have the right reference
                        async with prompt_server.distributed_jobs_lock:
                            q = prompt_server.distributed_pending_jobs[multi_job_id]
                            current_size = q.qsize()
                        
                        result = await asyncio.wait_for(q.get(), timeout=slice_timeout)
                        worker_id = result['worker_id']
                        is_last = result.get('is_last', False)
                        
                        # Check if batch mode
                        tensors = result.get('tensors', [])
                        indices = result.get('indices', [])  # Get the indices
                        if tensors:
                            # Batch mode
                            debug_log(f"Master - Got batch from worker {worker_id}, size={len(tensors)}, is_last={is_last}")
                            
                            if worker_id not in worker_images:
                                worker_images[worker_id] = {}
                            
                            # Use actual indices if available, otherwise fall back to sequential
                            if indices:
                                for i, tensor in enumerate(tensors):
                                    actual_idx = indices[i]
                                    worker_images[worker_id][actual_idx] = tensor
                            else:
                                # Fallback for backward compatibility
                                for idx, tensor in enumerate(tensors):
                                    worker_images[worker_id][idx] = tensor
                            
                            collected_count += len(tensors)
                        else:
                            # Single image mode (backward compat)
                            image_index = result['image_index']
                            tensor = result['tensor']
                            
                            debug_log(f"Master - Got single result from worker {worker_id}, image {image_index}, is_last={is_last}")
                            
                            if worker_id not in worker_images:
                                worker_images[worker_id] = {}
                            worker_images[worker_id][image_index] = tensor
                            
                            collected_count += 1
                        
                        # Record activity and refresh timeout baseline
                        last_activity = time.time()
                        base_timeout = float(get_worker_timeout_seconds())
                        
                        if is_last:
                            workers_done.add(worker_id)
                            p.update(1)  # +1 per completed worker
                        
                    except asyncio.TimeoutError:
                        # If we still have time, continue polling; otherwise handle timeout
                        if (time.time() - last_activity) < base_timeout:
                            comfy.model_management.throw_exception_if_processing_interrupted()
                            continue
                        # Re-check for user interruption after timeout expiry
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        missing_workers = set(str(w) for w in enabled_workers) - workers_done
                        log(f"Master - Timeout. Still waiting for workers: {list(missing_workers)}")

                        # Probe missing workers' /prompt endpoints to check if they are actively processing
                        any_busy = False
                        try:
                            cfg = load_config()
                            cfg_workers = cfg.get('workers', [])
                            session = await get_client_session()
                            for wid in list(missing_workers):
                                wrec = next((w for w in cfg_workers if str(w.get('id')) == str(wid)), None)
                                if not wrec:
                                    debug_log(f"Collector probe: worker {wid} not found in config")
                                    continue
                                host = wrec.get('host') or 'localhost'
                                port = int(wrec.get('port', 8188))
                                url = f"http://{host}:{port}/prompt"
                                try:
                                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                                        status = resp.status
                                        q = None
                                        if status == 200:
                                            try:
                                                payload = await resp.json()
                                                q = int(payload.get('exec_info', {}).get('queue_remaining', 0))
                                            except Exception:
                                                q = 0
                                        debug_log(f"Collector probe: worker {wid} status={status} queue_remaining={q}")
                                        if status == 200 and q and q > 0:
                                            any_busy = True
                                            log(f"Master - Probe grace: worker {wid} appears busy (queue_remaining={q}). Continuing to wait.")
                                            break
                                except Exception as e:
                                    debug_log(f"Collector probe failed for worker {wid}: {e}")
                        except Exception as e:
                            debug_log(f"Collector probe setup error: {e}")

                        if any_busy:
                            # Refresh last_activity and continue waiting
                            last_activity = time.time()
                            # Refresh base timeout in case the user changed it in UI
                            base_timeout = float(get_worker_timeout_seconds())
                            continue
                        
                        # Check queue size again with lock
                        async with prompt_server.distributed_jobs_lock:
                            if multi_job_id in prompt_server.distributed_pending_jobs:
                                final_q = prompt_server.distributed_pending_jobs[multi_job_id]
                                final_size = final_q.qsize()
                                
                                # Try to drain any remaining items
                                remaining_items = []
                                while not final_q.empty():
                                    try:
                                        item = final_q.get_nowait()
                                        remaining_items.append(item)
                                    except asyncio.QueueEmpty:
                                        break
                                
                                if remaining_items:
                                    # Process them
                                    for item in remaining_items:
                                        worker_id = item['worker_id']
                                        is_last = item.get('is_last', False)
                                        
                                        # Check if batch mode
                                        tensors = item.get('tensors', [])
                                        if tensors:
                                            # Batch mode
                                            if worker_id not in worker_images:
                                                worker_images[worker_id] = {}
                                            
                                            for idx, tensor in enumerate(tensors):
                                                worker_images[worker_id][idx] = tensor
                                            
                                            collected_count += len(tensors)
                                        else:
                                            # Single image mode
                                            image_index = item['image_index']
                                            tensor = item['tensor']
                                            
                                            if worker_id not in worker_images:
                                                worker_images[worker_id] = {}
                                            worker_images[worker_id][image_index] = tensor
                                            
                                            collected_count += 1
                                        
                                        if is_last:
                                            workers_done.add(worker_id)
                                            p.update(1)  # +1 here too
                            else:
                                log(f"Master - Queue {multi_job_id} no longer exists!")
                        break
            except comfy.model_management.InterruptProcessingException:
                # Cleanup queue on interruption and re-raise to abort prompt cleanly
                async with prompt_server.distributed_jobs_lock:
                    if multi_job_id in prompt_server.distributed_pending_jobs:
                        del prompt_server.distributed_pending_jobs[multi_job_id]
                raise
            
            total_collected = sum(len(imgs) for imgs in worker_images.values())
            
            # Clean up job queue
            async with prompt_server.distributed_jobs_lock:
                if multi_job_id in prompt_server.distributed_pending_jobs:
                    del prompt_server.distributed_pending_jobs[multi_job_id]

            # Reorder images according to seed distribution pattern
            # Pattern: master img 1, master img 2, worker 1 img 1, worker 1 img 2, worker 2 img 1, worker 2 img 2, etc.
            ordered_tensors = []
            
            # Add master images first (if any)
            if not delegate_mode and images_on_cpu is not None:
                for i in range(master_batch_size):
                    ordered_tensors.append(images_on_cpu[i:i+1])
            
            # Add worker images in order
            # The worker IDs in worker_images are already strings (e.g., "1", "2")
            # Just iterate through what we actually received
            for worker_id_str in sorted(worker_images.keys()):
                # Sort by image index for each worker
                for idx in sorted(worker_images[worker_id_str].keys()):
                    ordered_tensors.append(worker_images[worker_id_str][idx])
            
            # Ensure all tensors are on CPU and properly formatted before concatenation
            cpu_tensors = []
            for t in ordered_tensors:
                if t.is_cuda:
                    t = t.cpu()
                # Ensure tensor is contiguous in memory
                t = ensure_contiguous(t)
                cpu_tensors.append(t)
            
            try:
                if cpu_tensors:
                    combined = torch.cat(cpu_tensors, dim=0)
                else:
                    # No tensors collected (likely delegate mode with no worker output)
                    combined = ensure_contiguous(images) if images is not None else images
                    if combined is None:
                        raise ValueError("No image data collected from master or workers")
                # Ensure the combined tensor is contiguous and properly formatted
                combined = ensure_contiguous(combined)
                debug_log(f"Master - Job {multi_job_id} complete. Combined {combined.shape[0]} images total (master: {master_batch_size}, workers: {combined.shape[0] - master_batch_size})")
                return (combined,)
            except Exception as e:
                log(f"Master - Error combining images: {e}")
                # Return just the master images as fallback
                return (images,)

# --- Distributor Node ---
class DistributedSeed:
    """
    Distributes seed values across multiple GPUs.
    On master: passes through the original seed.
    On workers: adds offset based on worker ID.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 1125899906842, 
                    "min": 0,
                    "max": 1125899906842624,
                    "forceInput": False  # Widget by default, can be converted to input
                }),
            },
            "hidden": {
                "is_worker": ("BOOLEAN", {"default": False}),
                "worker_id": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "distribute"
    CATEGORY = "utils"
    
    def distribute(self, seed, is_worker=False, worker_id=""):
        if not is_worker:
            # Master node: pass through original values
            debug_log(f"Distributor - Master: seed={seed}")
            return (seed,)
        else:
            # Worker node: apply offset based on worker index
            # Find worker index from enabled_worker_ids
            try:
                # Worker IDs are passed as "worker_0", "worker_1", etc.
                if worker_id.startswith("worker_"):
                    worker_index = int(worker_id.split("_")[1])
                else:
                    # Fallback: try to parse as direct index
                    worker_index = int(worker_id)
                
                offset = worker_index + 1
                new_seed = seed + offset
                debug_log(f"Distributor - Worker {worker_index}: seed={seed} → {new_seed}")
                return (new_seed,)
            except (ValueError, IndexError) as e:
                debug_log(f"Distributor - Error parsing worker_id '{worker_id}': {e}")
                # Fallback: return original seed
                return (seed,)

# Define ByPassTypeTuple for flexible return types
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return any_type
        return item

class ImageBatchDivider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "divide_by": ("INT", {
                    "default": 2, 
                    "min": 1, 
                    "max": 10, 
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of parts to divide the batch into"
                }),
            }
        }
    
    RETURN_TYPES = ByPassTypeTuple(("IMAGE", ))  # Flexible for variable outputs
    RETURN_NAMES = ByPassTypeTuple(tuple([f"batch_{i+1}" for i in range(10)]))
    FUNCTION = "divide_batch"
    OUTPUT_NODE = True
    CATEGORY = "image"
    
    def divide_batch(self, images, divide_by):
        import torch
        
        # Use divide_by directly
        total_splits = divide_by
        
        if total_splits > 10:
            total_splits = 10  # Cap to max
        
        # Get total number of frames
        total_frames = images.shape[0]
        frames_per_split = total_frames // total_splits
        remainder = total_frames % total_splits
        
        outputs = []
        start_idx = 0
        
        for i in range(total_splits):
            current_frames = frames_per_split + (1 if i < remainder else 0)
            end_idx = start_idx + current_frames
            split_frames = images[start_idx:end_idx]
            outputs.append(split_frames)
            start_idx = end_idx
        
        # Pad with empty tensors up to max (10) to match potential RETURN_TYPES len
        empty_shape = (1, images.shape[1], images.shape[2], images.shape[3]) if total_frames > 0 else (1, 512, 512, 3)
        empty_tensor = torch.zeros(empty_shape, dtype=images.dtype if total_frames > 0 else torch.float32, 
                                   device=images.device if total_frames > 0 else 'cpu')
        
        while len(outputs) < 10:
            outputs.append(empty_tensor)
        
        return tuple(outputs)


class DistributedEmptyImage:
    """Produces an empty IMAGE batch used when the master delegates all work."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "width": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "channels": ("INT", {"default": 3, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create"
    CATEGORY = "image"

    def create(self, height, width, channels):
        import torch

        shape = (0, height, width, channels)
        tensor = torch.zeros(shape, dtype=torch.float32)
        return (tensor,)


NODE_CLASS_MAPPINGS = { 
    "DistributedCollector": DistributedCollectorNode,
    "DistributedSeed": DistributedSeed,
    "ImageBatchDivider": ImageBatchDivider,
    "DistributedEmptyImage": DistributedEmptyImage,
}
NODE_DISPLAY_NAME_MAPPINGS = { 
    "DistributedCollector": "Distributed Collector",
    "DistributedSeed": "Distributed Seed",
    "ImageBatchDivider": "Image Batch Divider",
    "DistributedEmptyImage": "Distributed Empty Image",
}
