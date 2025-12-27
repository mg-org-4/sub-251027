# gguf_quantizer_node.py
import os
import subprocess
import shutil
import sys
import platform
import tempfile
import uuid # For unique temporary file names
from safetensors.torch import save_file # For saving the model state_dict

# ComfyUI imports
try:
    import folder_paths
    # import comfy.model_management # For type checking or detailed inspection if needed
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    # Fallback for paths if ComfyUI is not fully available
    class folder_paths:
        @staticmethod
        def get_input_directory(): return os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
        @staticmethod
        def get_output_directory(): return os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        @staticmethod
        def get_temp_directory(): return os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        @staticmethod
        def get_folder_paths(folder_name): return [os.path.join(folder_paths.get_input_directory(), folder_name)]
        @staticmethod
        def get_filename_list(folder_name): # Not directly used by this node version
            pass


# --- GGUFImageQuantizer Core Logic ---
class GGUFImageQuantizer:
    def __init__(self, base_node_dir: str, verbose: bool = True):
        self.base_node_dir = base_node_dir
        self.verbose = verbose
        self.llama_cpp_src_dir = os.path.join(self.base_node_dir, "llama_cpp_src")
        
        self.quantize_exe_name = "llama-quantize.exe" if platform.system() == "Windows" else "llama-quantize"
        
        # Initial path guess, might be refined after build
        self.compiled_quantize_exe_path = os.path.join(
            self.llama_cpp_src_dir, "build", "bin", self.quantize_exe_name
        )
        if platform.system() == "Windows" and not os.path.exists(self.compiled_quantize_exe_path):
             self.compiled_quantize_exe_path = os.path.join(
                self.llama_cpp_src_dir, "build", "bin", "Release", self.quantize_exe_name
            )

        gguf_scripts_subdir = "gguf" # Assumes a 'gguf' subdir in the node's directory for these scripts
        self.convert_script = os.path.join(self.base_node_dir, gguf_scripts_subdir, "convert.py")
        self.fix_5d_script = os.path.join(self.base_node_dir, gguf_scripts_subdir, "fix_5d_tensors.py")
        self.patch_file = os.path.join(self.base_node_dir, gguf_scripts_subdir, "lcpp.patch")
        self.fix_lines_script = os.path.join(self.base_node_dir, gguf_scripts_subdir, "fix_lines_ending.py")

        self.current_model_arch = None
        if self.verbose:
            print("DEBUG: GGUFImageQuantizer initialized.")

    def _get_python_executable(self):
        if self.verbose:
            print("DEBUG: _get_python_executable called.")
        return sys.executable if sys.executable else "python"

    def _check_cmake_availability(self):
        """Check if CMake is available on the system using direct subprocess call."""
        print("[GGUF Image Quantizer] 🔍 Checking CMake availability...")

        try:
            # Use subprocess.run directly for more reliable checking
            result = subprocess.run(
                ["cmake", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("[GGUF Image Quantizer] ✅ CMake is available and working!")
                if self.verbose and result.stdout:
                    # Show first line of version info
                    version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown version"
                    print(f"[GGUF Image Quantizer] CMake version: {version_line}")
                return True
            else:
                print("[GGUF Image Quantizer] ❌ CMake command failed!")
                if result.stderr:
                    print(f"[GGUF Image Quantizer] CMake error: {result.stderr}")
                return False

        except FileNotFoundError:
            print("[GGUF Image Quantizer] ❌ CMake not found! CMake is not installed or not in system PATH.")
            return False
        except subprocess.TimeoutExpired:
            print("[GGUF Image Quantizer] ❌ CMake check timed out!")
            return False
        except Exception as e:
            print(f"[GGUF Image Quantizer] ❌ Unexpected error checking CMake: {e}")
            return False

    def _print_cmake_installation_instructions(self):
        """Print platform-specific CMake installation instructions."""
        print("\n" + "="*80)
        print("❌ CRITICAL ERROR: CMake is not installed or not found in system PATH")
        print("="*80)
        print("🔧 GGUF Quantizer requires CMake to build llama.cpp quantization tools.")
        print("📍 This is an updated version with improved CMake detection.")
        print("\nPlease install CMake using one of these methods:\n")

        system = platform.system()
        if system == "Windows":
            print("🪟 WINDOWS:")
            print("  Option 1: Download from official website")
            print("    • Visit: https://cmake.org/download/")
            print("    • Download the Windows installer (.msi)")
            print("    • Run installer and make sure to check 'Add CMake to system PATH'\n")
            print("  Option 2: Using Chocolatey")
            print("    • Run: choco install cmake\n")
            print("  Option 3: Using winget")
            print("    • Run: winget install Kitware.CMake\n")
        elif system == "Darwin":
            print("🍎 macOS:")
            print("  Option 1: Using Homebrew (recommended)")
            print("    • Run: brew install cmake\n")
            print("  Option 2: Download from official website")
            print("    • Visit: https://cmake.org/download/")
            print("    • Download the macOS installer (.dmg)\n")
        else:  # Linux and others
            print("🐧 LINUX:")
            print("  Ubuntu/Debian:")
            print("    • Run: sudo apt update && sudo apt install cmake\n")
            print("  RHEL/CentOS/Fedora:")
            print("    • Run: sudo yum install cmake  (or sudo dnf install cmake)\n")
            print("  Arch Linux:")
            print("    • Run: sudo pacman -S cmake\n")

        print("After installing CMake:")
        print("  1. Restart your terminal/command prompt")
        print("  2. Verify installation by running: cmake --version")
        print("  3. Try running the GGUF quantizer node again")
        print("="*80 + "\n")

    def _run_subprocess(self, command: list, cwd: str = None, desc: str = ""):
        if desc and self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: Running: {desc} (Command: {' '.join(command)}) (CWD: {cwd if cwd else 'None'})")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            stdout, stderr = process.communicate(timeout=600) # 10 minute timeout

            # --- START VERBOSE MODIFICATION ---
            if self.verbose and stdout and stdout.strip(): # Only print if there's actual output
                print(f"[GGUF Image Quantizer] STDOUT from '{desc}':\n{stdout.strip()}")
            if stderr and stderr.strip(): # Always print stderr, even if returncode is 0, as it might contain warnings
                print(f"[GGUF Image Quantizer] STDERR from '{desc}':\n{stderr.strip()}")
            # --- END VERBOSE MODIFICATION ---

            if process.returncode != 0:
                print(f"[GGUF Image Quantizer] Error during '{desc}' (Return Code: {process.returncode}). See STDERR above if any.")
                return False, stdout, stderr
            
            if self.verbose:
                print(f"[GGUF Image Quantizer] Success: {desc}")
            return True, stdout, stderr
        except subprocess.TimeoutExpired:
            print(f"[GGUF Image Quantizer] Timeout during '{desc}' after 10 minutes.")
            return False, "", "TimeoutExpired"
        except Exception as e:
            print(f"[GGUF Image Quantizer] Exception during '{desc}': {e}")
            if self.verbose:
                import traceback
                print(f"DEBUG: Traceback for _run_subprocess exception: {traceback.format_exc()}")
            return False, "", str(e)

    def setup_llama_cpp(self):
        if self.verbose:
            print("[GGUF Image Quantizer] DEBUG: Starting setup_llama_cpp...")
        os.makedirs(self.llama_cpp_src_dir, exist_ok=True)
        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: Ensured llama_cpp_src_dir exists: {self.llama_cpp_src_dir}")

        gguf_scripts_dir = os.path.join(self.base_node_dir, "gguf")
        if os.path.exists(self.fix_lines_script) and os.path.exists(self.patch_file):
            if self.verbose:
                print("[GGUF Image Quantizer] DEBUG: Found fix_lines_script and patch_file. Attempting to fix line endings for patch file.")
            self._run_subprocess(
                [self._get_python_executable(), self.fix_lines_script, self.patch_file],
                cwd=gguf_scripts_dir, 
                desc="Fix patch file line endings"
            )
        else:
            if self.verbose:
                print(f"[GGUF Image Quantizer] DEBUG: Skipping fix line endings. fix_lines_script exists: {os.path.exists(self.fix_lines_script)}, patch_file exists: {os.path.exists(self.patch_file)}")


        git_repo_path = os.path.join(self.llama_cpp_src_dir, ".git")
        if not os.path.exists(git_repo_path):
            if self.verbose:
                print(f"[GGUF Image Quantizer] DEBUG: .git directory not found at {git_repo_path}. Cloning llama.cpp.")
            success, _, _ = self._run_subprocess(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", self.llama_cpp_src_dir],
                desc="Clone llama.cpp"
            )
            if not success: 
                if self.verbose:
                    print("[GGUF Image Quantizer] DEBUG: Cloning llama.cpp failed.")
                return False
        else:
            if self.verbose:
                print("[GGUF Image Quantizer] DEBUG: llama.cpp repository already cloned. Fetching updates...")
            self._run_subprocess(["git", "fetch", "--tags"], cwd=self.llama_cpp_src_dir, desc="Git fetch llama.cpp tags")
        
        readme_checkout_tag = "b3962" 
        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: Checking out llama.cpp tag: {readme_checkout_tag}...")
        success, _, _ = self._run_subprocess(
            ["git", "checkout", f"tags/{readme_checkout_tag}"], cwd=self.llama_cpp_src_dir, desc=f"Checkout tag {readme_checkout_tag}"
        )
        if not success:
            if self.verbose:
                print(f"[GGUF Image Quantizer] DEBUG: Failed to checkout tag {readme_checkout_tag}. Trying git pull and re-checkout.")
            self._run_subprocess(["git", "pull"], cwd=self.llama_cpp_src_dir, desc="Git pull after failed checkout")
            success, _, _ = self._run_subprocess(
                 ["git", "checkout", f"tags/{readme_checkout_tag}"], cwd=self.llama_cpp_src_dir, desc=f"Retry checkout tag {readme_checkout_tag}"
            )
            if not success:
                if self.verbose:
                    print(f"[GGUF Image Quantizer] DEBUG: Critical: Failed to checkout required llama.cpp tag {readme_checkout_tag}. Patching and compilation may fail.")
                # return False # Or allow to proceed with caution

        patch_check_file = os.path.join(self.llama_cpp_src_dir, "gguf-py", "gguf", "constants.py") 
        patch_applied_sentinel = "LLM_ARCH_FLUX" 
        
        already_applied = False
        if os.path.exists(patch_check_file):
            try:
                print(f"DEBUG: Checking for patch sentinel '{patch_applied_sentinel}' in {patch_check_file}")
                with open(patch_check_file, 'r', encoding='utf-8', errors='ignore') as f_check:
                    if patch_applied_sentinel in f_check.read():
                        already_applied = True
                        print("[GGUF Image Quantizer] DEBUG: Patch sentinel found. Assuming patch is applied.")
            except Exception as e:
                 print(f"[GGUF Image Quantizer] DEBUG: Warning: Could not check if patch was applied due to: {e}")
        else:
            print(f"DEBUG: Patch check file {patch_check_file} does not exist.")

        
        if not already_applied:
            if self.verbose:
                print("[GGUF Image Quantizer] DEBUG: Patch not detected as applied.")
            if not os.path.exists(self.patch_file):
                print(f"[GGUF Image Quantizer] DEBUG: Error: Patch file not found at {self.patch_file}. Cannot apply patch.")
                return False
            
            if self.verbose:
                print("[GGUF Image Quantizer] DEBUG: Attempting to reverse any existing patches (best effort)...")
            self._run_subprocess( 
                ["git", "apply", "--reverse", "--reject", self.patch_file],
                cwd=self.llama_cpp_src_dir,
                desc="Reverse existing patches" 
            )
            if self.verbose:
                print("[GGUF Image Quantizer] DEBUG: Applying lcpp.patch...")
            success, stdout_patch, stderr_patch = self._run_subprocess(
                ["git", "apply", "--ignore-whitespace", self.patch_file], 
                cwd=self.llama_cpp_src_dir,
                desc="Apply lcpp.patch"
            )
            if not success:
                if self.verbose:
                    print(f"[GGUF Image Quantizer] DEBUG: Failed to apply patch.")
                if os.path.exists(patch_check_file):
                    if self.verbose:
                        print(f"DEBUG: Re-checking for patch sentinel in {patch_check_file} after failed apply.")
                    with open(patch_check_file, 'r', encoding='utf-8', errors='ignore') as f_check_after_fail:
                        if patch_applied_sentinel in f_check_after_fail.read():
                            if self.verbose:
                                print("[GGUF Image Quantizer] DEBUG: Patch sentinel FOUND despite 'git apply' error. Proceeding cautiously.")
                        else:
                            if self.verbose:
                                print("[GGUF Image Quantizer] DEBUG: Patch sentinel NOT FOUND after 'git apply' error. Setup failed.")
                            return False
                else:
                     if self.verbose:
                         print(f"[GGUF Image Quantizer] DEBUG: Patch check file {patch_check_file} not found after 'git apply' error. Setup failed.")
                     return False
        else:
            if self.verbose:
                print("[GGUF Image Quantizer] DEBUG: Patch already applied or sentinel found. Skipping patch application.")

        # Check if CMake is available before proceeding with build
        if not self._check_cmake_availability():
            self._print_cmake_installation_instructions()
            return False

        build_dir = os.path.join(self.llama_cpp_src_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: Ensured build directory exists: {build_dir}")

        cmake_cache_file = os.path.join(build_dir, "CMakeCache.txt")
        if not os.path.exists(cmake_cache_file):
            if self.verbose:
                print("[GGUF Image Quantizer] DEBUG: CMakeCache.txt not found. Configuring CMake for llama-quantize (CPU build)...")

            # Double-check CMake availability before running cmake command
            if not self._check_cmake_availability():
                print("[GGUF Image Quantizer] ❌ CRITICAL: CMake check failed just before configuration!")
                self._print_cmake_installation_instructions()
                return False

            cmake_cmd = ["cmake", "..", "-DLLAMA_ACCELERATE=OFF", "-DLLAMA_METAL=OFF", "-DLLAMA_CUDA=OFF", "-DLLAMA_VULKAN=OFF", "-DLLAMA_SYCL=OFF", "-DLLAMA_OPENCL=OFF", "-DLLAMA_BLAS=OFF", "-DLLAMA_LAPACK=OFF"]
            success, _, _ = self._run_subprocess(cmake_cmd, cwd=build_dir, desc="CMake configuration")
            if not success:
                if self.verbose:
                    print("[GGUF Image Quantizer] DEBUG: CMake configuration failed.")
                print("[GGUF Image Quantizer] 💡 If you see 'file not found' errors above, CMake might not be properly installed.")
                self._print_cmake_installation_instructions()
                return False
        else:
            if self.verbose:
                print("[GGUF Image Quantizer] DEBUG: CMake cache found. Assuming already configured. Skipping CMake configuration.")

        if self.verbose:
            print("[GGUF Image Quantizer] DEBUG: Building llama-quantize target...")

        # Final CMake check before build
        if not self._check_cmake_availability():
            print("[GGUF Image Quantizer] ❌ CRITICAL: CMake check failed just before build!")
            self._print_cmake_installation_instructions()
            return False

        cmake_build_cmd = ["cmake", "--build", ".", "--target", "llama-quantize"]
        if platform.system() == "Windows":
             cmake_build_cmd.extend(["--config", "Release"])

        success, _, _ = self._run_subprocess(cmake_build_cmd, cwd=build_dir, desc="CMake build llama-quantize")
        if not success:
            if self.verbose:
                print("[GGUF Image Quantizer] DEBUG: CMake build llama-quantize failed.")
            print("[GGUF Image Quantizer] 💡 If you see 'file not found' errors above, CMake might not be properly installed.")
            self._print_cmake_installation_instructions()
            return False
        
        self.compiled_quantize_exe_path = os.path.join(self.llama_cpp_src_dir, "build", "bin", self.quantize_exe_name)
        if platform.system() == "Windows" and not os.path.exists(self.compiled_quantize_exe_path):
            self.compiled_quantize_exe_path = os.path.join(self.llama_cpp_src_dir, "build", "bin", "Release", self.quantize_exe_name)
        
        if not os.path.exists(self.compiled_quantize_exe_path):
            alt_path = os.path.join(build_dir, self.quantize_exe_name)
            if os.path.exists(alt_path):
                self.compiled_quantize_exe_path = alt_path
                if self.verbose:
                    print(f"[GGUF Image Quantizer] DEBUG: Found llama-quantize at alternate path: {alt_path}")
            else:
                if self.verbose:
                    print(f"[GGUF Image Quantizer] DEBUG: Compiled llama-quantize not found at expected paths after build.")
                return False
        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: llama-quantize path set to: {self.compiled_quantize_exe_path}")

        if self.verbose:
            print("[GGUF Image Quantizer] DEBUG: llama.cpp environment setup complete.")
        return True

    def convert_model_to_initial_gguf(self, model_src_path: str, temp_conversion_dir: str):
        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: Starting convert_model_to_initial_gguf. Src: {model_src_path}, TempDir: {temp_conversion_dir}")
        if not os.path.exists(self.convert_script):
            print(f"DEBUG: Error: GGUF convert.py script not found at {self.convert_script}")
            return None, None

        base_name = os.path.splitext(os.path.basename(model_src_path))[0]
        # Specify the exact output path to avoid filename confusion
        expected_gguf_name_f16 = f"{base_name}-F16.gguf"
        expected_output_path = os.path.join(temp_conversion_dir, expected_gguf_name_f16)
        cmd = [self._get_python_executable(), self.convert_script, "--src", model_src_path, "--dst", expected_output_path]

        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: About to run convert.py. Command: {' '.join(cmd)}")
        success, stdout, stderr = self._run_subprocess(
            cmd,
            cwd=temp_conversion_dir,
            desc="Convert model to initial GGUF (FP16)"
        )
        if not success:
            if self.verbose:
                print(f"[GGUF Image Quantizer] DEBUG: convert.py execution failed.")
            return None, None

        initial_gguf_path = None
        model_arch = None

        for line in stdout.splitlines():
            line_lower = line.lower()
            if "model architecture:" in line_lower:
                model_arch = line_lower.split("model architecture:")[-1].strip()
                if self.verbose:
                    print(f"DEBUG: Parsed model_arch (from 'model architecture:'): {model_arch}")
                break
            elif "llm_arch =" in line:
                model_arch = line.split("=")[-1].strip().replace("'", "").replace('"',"")
                if self.verbose:
                    print(f"DEBUG: Parsed model_arch (from 'llm_arch ='): {model_arch}")
                break

        # Check if the file was created at the expected path (which we specified with --dst)
        if os.path.exists(expected_output_path):
            initial_gguf_path = expected_output_path
            if self.verbose:
                print(f"DEBUG: Found initial GGUF at expected path: {initial_gguf_path}")
        else:
            if self.verbose:
                print(f"DEBUG: Expected GGUF file not found at {expected_output_path}. Scanning directory {temp_conversion_dir}...")
            for fname in os.listdir(temp_conversion_dir):
                if fname.lower().endswith(".gguf"):
                    initial_gguf_path = os.path.join(temp_conversion_dir, fname)
                    if self.verbose:
                        print(f"[GGUF Image Quantizer] DEBUG: Found GGUF file by scan: {fname}")
                    break

        if not initial_gguf_path:
            if self.verbose:
                print(f"[GGUF Image Quantizer] DEBUG: Could not find the output GGUF file in {temp_conversion_dir}.")
            return None, None

        if model_arch:
            self.current_model_arch = model_arch.lower()
            if self.verbose:
                print(f"[GGUF Image Quantizer] DEBUG: Detected model architecture from script output: '{self.current_model_arch}'")
        else:
            if self.verbose:
                print("DEBUG: Model architecture not found in script output. Attempting to guess from filename.")
            fn_lower = os.path.basename(initial_gguf_path).lower()
            if "clip" in fn_lower: self.current_model_arch = "clip"
            elif "siglip" in fn_lower: self.current_model_arch = "siglip"
            elif "flux" in fn_lower: self.current_model_arch = "flux"

            if self.current_model_arch:
                if self.verbose:
                    print(f"[GGUF Image Quantizer] DEBUG: Guessed model architecture from filename: '{self.current_model_arch}'")
            else:
                if self.verbose:
                    print("[GGUF Image Quantizer] DEBUG: Warning: Model architecture could not be determined.")

        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: Initial GGUF created at: {initial_gguf_path}. Architecture: {self.current_model_arch}")
        return initial_gguf_path, self.current_model_arch


    def quantize_gguf(self, initial_gguf_path_in_temp: str, quant_type: str, final_output_gguf_path: str):
        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: Starting quantize_gguf. Initial: {initial_gguf_path_in_temp}, Type: {quant_type}, Final: {final_output_gguf_path}")
        if not os.path.exists(self.compiled_quantize_exe_path):
            print(f"DEBUG: Error: Compiled llama-quantize not found at {self.compiled_quantize_exe_path}")
            return None
        if not os.path.exists(initial_gguf_path_in_temp):
            print(f"DEBUG: Error: Initial GGUF file not found for quantization: {initial_gguf_path_in_temp}")
            return None

        os.makedirs(os.path.dirname(final_output_gguf_path), exist_ok=True)
        if self.verbose:
            print(f"DEBUG: Ensured output directory for quantized file exists: {os.path.dirname(final_output_gguf_path)}")

        cmd = [self.compiled_quantize_exe_path, initial_gguf_path_in_temp, final_output_gguf_path, quant_type.upper()]
        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: About to run llama-quantize. Command: {' '.join(cmd)}")
        success, stdout_quant, stderr_quant = self._run_subprocess(cmd, desc=f"Convert/Quantize GGUF to {quant_type}")

        if not success or not os.path.exists(final_output_gguf_path):
            if self.verbose:
                print(f"[GGUF Image Quantizer] DEBUG: Failed to process to {quant_type} or output file not found: {final_output_gguf_path}")
            return None

        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: Successfully processed to {quant_type}: {final_output_gguf_path}")
        return final_output_gguf_path

    def apply_5d_fix_if_needed(self, target_final_gguf_path: str, model_arch: str, gguf_scripts_dir: str):
        if self.verbose:
            print(f"DEBUG: Starting apply_5d_fix_if_needed. Target: {target_final_gguf_path}, Arch: {model_arch}, ScriptsDir: {gguf_scripts_dir}")
        if not model_arch:
            if self.verbose:
                print("[GGUF Image Quantizer] DEBUG: No model architecture provided; skipping 5D tensor fix.")
            return target_final_gguf_path

        fix_safetensor_filename = f"fix_5d_tensors_{model_arch.lower()}.safetensors"
        fix_safetensor_path = os.path.join(gguf_scripts_dir, fix_safetensor_filename)
        if self.verbose:
            print(f"DEBUG: Expected 5D fix definition file path: {fix_safetensor_path}")

        if not os.path.exists(fix_safetensor_path):
            if self.verbose:
                print(f"[GGUF Image Quantizer] DEBUG: No 5D fix definition file found for arch '{model_arch}' at {fix_safetensor_path}. Skipping 5D fix.")
            return target_final_gguf_path

        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: Applying 5D tensor fix for model arch: {model_arch} using {fix_safetensor_path}")
        if not os.path.exists(self.fix_5d_script):
            print(f"DEBUG: Error: fix_5d_tensors.py script not found at {self.fix_5d_script}")
            return None
        if not os.path.exists(target_final_gguf_path):
             print(f"DEBUG: Error: Target GGUF for 5D fix not found: {target_final_gguf_path}")
             return None

        cmd = [self._get_python_executable(), self.fix_5d_script,
               "--src", target_final_gguf_path,
               "--dst", target_final_gguf_path,
               "--fix", fix_safetensor_path,
               "--overwrite"]

        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: About to run fix_5d_tensors.py. Command: {' '.join(cmd)}")
        success, stdout_fix, stderr_fix = self._run_subprocess(
            cmd,
            cwd=gguf_scripts_dir,
            desc="Apply 5D tensor fix"
        )
        if not success:
            if self.verbose:
                print(f"[GGUF Image Quantizer] DEBUG: Failed to apply 5D fix to {target_final_gguf_path}.")
            return None

        if self.verbose:
            print(f"[GGUF Image Quantizer] DEBUG: 5D tensor fix applied. Final model at: {target_final_gguf_path}")
        return target_final_gguf_path


# --- ComfyUI Node ---
class GGUFQuantizerNode:
    QUANT_TYPES = sorted(["F16", "BF16", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
                          "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS",
                          "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L"])

    @classmethod
    def INPUT_TYPES(cls):
        extended_quant_types = cls.QUANT_TYPES + ["ALL"]
        return {
            "required": {
                "model": ("MODEL",),
                "quantization_type": (extended_quant_types, {"default": "Q4_K_M"}),
                "output_path_template": ("STRING", {"default": "gguf_quantized/piped_model", "multiline": False, "placeholder": "folder/name_core OR /abs_path/folder/name_core"}),
                "is_absolute_path": ("BOOLEAN", {"default": False, "label_on": "Absolute Path Mode", "label_off": "Relative to ComfyUI Output Dir"}),
                "setup_environment": ("BOOLEAN", {"default": False, "label_on": "Run Setup First (llama.cpp)", "label_off": "Skip Setup (if already done)"}),
                "verbose_logging": ("BOOLEAN", {"default": True, "label_on": "Verbose Debug Logging", "label_off": "Minimal Logging"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("status_message", "output_gguf_path_or_dir",)
    FUNCTION = "quantize_diffusion_model"
    CATEGORY = "Model Quantization/GGUF"
    OUTPUT_NODE = True

    def quantize_diffusion_model(self, model, quantization_type: str,
                                 output_path_template: str, is_absolute_path: bool,
                                 setup_environment: bool, verbose_logging: bool):

        base_node_dir = os.path.dirname(os.path.abspath(__file__))
        quantizer = GGUFImageQuantizer(base_node_dir, verbose=verbose_logging)
        status_messages = ["DEBUG: Starting GGUF Image Quantization Node..."]
        if verbose_logging:
            print("DEBUG: quantize_diffusion_model called with parameters:")
            print(f"DEBUG:   quantization_type: {quantization_type}")
            print(f"DEBUG:   output_path_template: {output_path_template}")
            print(f"DEBUG:   is_absolute_path: {is_absolute_path}")
            print(f"DEBUG:   setup_environment: {setup_environment}")
            print(f"DEBUG:   verbose_logging: {verbose_logging}")


        if setup_environment:
            status_messages.append("DEBUG: Attempting llama.cpp environment setup...")
            if verbose_logging:
                print("DEBUG: Calling quantizer.setup_llama_cpp()")
            if not quantizer.setup_llama_cpp(): # This method now has its own DEBUG prints
                status_messages.append("❌ Error: llama.cpp environment setup failed. Check console.")
                if verbose_logging:
                    print("DEBUG: quantizer.setup_llama_cpp() returned False.")
                return ("\n".join(status_messages), "")
            status_messages.append("✅ llama.cpp environment setup successful.")
            if verbose_logging:
                print("DEBUG: quantizer.setup_llama_cpp() returned True.")
        elif not os.path.exists(quantizer.compiled_quantize_exe_path):
            status_messages.append(f"❌ Error: llama-quantize not found at '{quantizer.compiled_quantize_exe_path}' and setup was skipped. Run with 'setup_environment=True' at least once.")
            if verbose_logging:
                print(f"DEBUG: llama-quantize not found at {quantizer.compiled_quantize_exe_path} and setup_environment is False.")
            return ("\n".join(status_messages), "")
        else:
            if verbose_logging:
                print(f"DEBUG: Skipping llama.cpp setup. Found llama-quantize at {quantizer.compiled_quantize_exe_path}")


        temp_model_input_path = None
        derived_model_name_for_output = "piped_unet_model"

        try:
            if verbose_logging:
                print("DEBUG: Entering UNET state_dict extraction and model name determination block.")
            unet_state_dict = None

            if hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
                if verbose_logging:
                    print("DEBUG: Trying to extract state_dict from model.model")
                unet_state_dict = model.model.state_dict()
                status_messages.append("✅ Extracted UNET state_dict from model.model")
                if hasattr(model, 'model_config'):
                    m_config = model.model_config
                    name_src = getattr(m_config, 'filename', getattr(m_config, 'name', None))
                    if isinstance(name_src, str) and name_src.strip() and not any(x in name_src.lower() for x in ["unet.json", "config.json"]):
                        derived_model_name_for_output = os.path.splitext(os.path.basename(name_src))[0]
                    elif hasattr(m_config, 'original_config_path') and isinstance(getattr(m_config, 'original_config_path', None), str):
                         derived_model_name_for_output = os.path.splitext(os.path.basename(m_config.original_config_path))[0]
                if verbose_logging:
                    print(f"DEBUG: Path 1: derived_model_name_for_output = {derived_model_name_for_output}")

            elif hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'state_dict'):
                if verbose_logging:
                    print("DEBUG: Trying to extract state_dict from model.model.model")
                unet_state_dict = model.model.model.state_dict()
                status_messages.append("✅ Extracted UNET state_dict from model.model.model")
                m_config = getattr(model.model, 'model_config', getattr(model, 'model_config', None))
                if m_config:
                    name_src = getattr(m_config, 'filename', getattr(m_config, 'name', None))
                    if isinstance(name_src, str) and name_src.strip() and not any(x in name_src.lower() for x in ["unet.json", "config.json"]):
                        derived_model_name_for_output = os.path.splitext(os.path.basename(name_src))[0]
                if verbose_logging:
                    print(f"DEBUG: Path 2: derived_model_name_for_output = {derived_model_name_for_output}")

            elif hasattr(model, 'diffusion_model') and hasattr(model.diffusion_model, 'state_dict'):
                if verbose_logging:
                    print("DEBUG: Trying to extract state_dict from model.diffusion_model")
                unet_state_dict = model.diffusion_model.state_dict()
                status_messages.append("✅ Extracted UNET state_dict from model.diffusion_model")
                m_config = getattr(model, 'model_config', None)
                if not m_config and hasattr(model.diffusion_model, 'config'):
                    diffusers_conf = getattr(model.diffusion_model, 'config', None)
                    name_or_path = getattr(diffusers_conf, '_name_or_path', "")
                    if isinstance(name_or_path, str) and name_or_path.strip():
                         derived_model_name_for_output = os.path.basename(name_or_path)
                         derived_model_name_for_output = os.path.splitext(derived_model_name_for_output)[0] if not os.path.isdir(os.path.join(".", name_or_path)) else derived_model_name_for_output
                elif m_config:
                    name_src = getattr(m_config, 'filename', getattr(m_config, 'name', None))
                    if isinstance(name_src, str) and name_src.strip() and not any(x in name_src.lower() for x in ["unet.json", "config.json"]):
                         derived_model_name_for_output = os.path.splitext(os.path.basename(name_src))[0]
                if verbose_logging:
                    print(f"DEBUG: Path 3: derived_model_name_for_output = {derived_model_name_for_output}")

            elif hasattr(model, 'state_dict'):
                if verbose_logging:
                    print("DEBUG: Trying to extract state_dict directly from model object")
                unet_state_dict = model.state_dict()
                status_messages.append("✅ Extracted state_dict directly from input model object")
                direct_conf = getattr(model, 'config', getattr(model, 'model_config', None))
                if direct_conf:
                    name_or_path = getattr(direct_conf, '_name_or_path', getattr(direct_conf, 'filename', getattr(direct_conf, 'name', None)))
                    if isinstance(name_or_path, str) and name_or_path.strip() and not any(x in name_or_path.lower() for x in ["unet.json", "config.json"]):
                        derived_model_name_for_output = os.path.basename(name_or_path)
                        derived_model_name_for_output = os.path.splitext(derived_model_name_for_output)[0] if not os.path.isdir(os.path.join(".", name_or_path)) else derived_model_name_for_output
                if verbose_logging:
                    print(f"DEBUG: Path 4: derived_model_name_for_output = {derived_model_name_for_output}")

            if unet_state_dict is None:
                if verbose_logging:
                    print("DEBUG: UNET state_dict is None after all checks.")
                model_type_info = f"Type of input model: {type(model)}."
                model_attrs_str = ""
                try:
                    model_attrs_str = f"Non-callable attributes: {', '.join(sorted(attr for attr in dir(model) if not callable(getattr(model, attr, None)) and not attr.startswith('__')))}"
                except: model_attrs_str = "Could not inspect model attributes."
                error_msg = (
                    "❌ Error: Could not extract UNET state_dict. The input 'model' doesn't match known ComfyUI MODEL structures or provide a direct state_dict.\n"
                    f"{model_type_info}\n{model_attrs_str[:1500]}"
                )
                status_messages.append(error_msg)
                return ("\n".join(status_messages), "") # Critical error, return

            status_messages.append(f"Using derived base name for output files: '{derived_model_name_for_output}'")
            if verbose_logging:
                print(f"DEBUG: Final derived_model_name_for_output: {derived_model_name_for_output}")

            temp_dir_for_input_model_sf = folder_paths.get_temp_directory()
            os.makedirs(temp_dir_for_input_model_sf, exist_ok=True)
            temp_model_input_path = os.path.join(temp_dir_for_input_model_sf, f"temp_unet_{derived_model_name_for_output}_{uuid.uuid4()}.safetensors")

            if verbose_logging:
                print(f"DEBUG: About to save UNET state_dict to temporary file: {temp_model_input_path}")
            save_file(unet_state_dict, temp_model_input_path)
            status_messages.append(f"✅ UNET state_dict saved to temporary file: {os.path.basename(temp_model_input_path)}")
            if verbose_logging:
                print(f"DEBUG: UNET state_dict saved successfully.")
            src_model_path_for_convert = temp_model_input_path

        except Exception as e:
            if verbose_logging:
                print(f"DEBUG: Exception during UNET state_dict extraction or saving: {e}")
            if temp_model_input_path and os.path.exists(temp_model_input_path):
                try: os.remove(temp_model_input_path)
                except: pass
            import traceback
            tb_str = traceback.format_exc()
            if verbose_logging:
                print(f"DEBUG: Traceback for state_dict exception: {tb_str}")
            status_messages.append(f"❌ Error during UNET state_dict extraction or saving: {e}\n{tb_str}")
            return ("\n".join(status_messages), "")

        status_messages.append(f"Preparing to convert & quantize using temporary UNET: {src_model_path_for_convert}")
        if verbose_logging:
            print(f"DEBUG: src_model_path_for_convert is set to: {src_model_path_for_convert}")

        # --- Determine Final Output Directory and Filename Core ---
        if verbose_logging:
            print("DEBUG: Starting output path determination block.")
        path_template_str = output_path_template.strip()
        filename_core = derived_model_name_for_output
        output_directory_part = ""

        if not path_template_str:
            if verbose_logging:
                print("DEBUG: output_path_template is empty.")
            if is_absolute_path:
                status_messages.append("❌ Error: 'output_path_template' cannot be empty when 'is_absolute_path' is True.")
                if verbose_logging:
                    print("DEBUG: Error - output_path_template empty in absolute_path mode.")
                if src_model_path_for_convert and os.path.exists(src_model_path_for_convert): os.remove(src_model_path_for_convert)
                return ("\n".join(status_messages), "")
            else:
                output_directory_part = "gguf_quantized"
                final_output_directory = os.path.join(folder_paths.get_output_directory(), output_directory_part)
                if verbose_logging:
                    print(f"DEBUG: Relative mode, empty template. Subdir: {output_directory_part}, Full dir: {final_output_directory}")
        else:
            if verbose_logging:
                print(f"DEBUG: output_path_template provided: '{path_template_str}'")
            norm_template = os.path.normpath(path_template_str)
            user_basename = os.path.basename(norm_template)
            user_dirname = os.path.dirname(norm_template)
            if verbose_logging:
                print(f"DEBUG: norm_template: {norm_template}, user_basename: {user_basename}, user_dirname: {user_dirname}")

            # Check if the path template is a directory path or a file path
            # If it's a directory path (ends with separator, or basename has no extension and looks like a folder name),
            # use the entire path as the directory and keep the original filename_core
            is_directory_path = (
                path_template_str.endswith(os.path.sep) or
                path_template_str.endswith('/') or
                (user_basename and
                 not '.' in user_basename and
                 len(user_basename) > 0 and
                 # Common directory names or patterns that suggest it's a directory
                 (user_basename.lower() in ['models', 'unet', 'checkpoints', 'gguf', 'output', 'quantized'] or
                  user_basename.lower().endswith('_models') or
                  user_basename.lower().endswith('_output') or
                  # If the parent directory exists and this looks like a subdirectory
                  (user_dirname and os.path.exists(user_dirname))))
            )

            if is_directory_path:
                # This is a directory path
                output_directory_part = norm_template
                # Keep the original filename_core (derived_model_name_for_output)
                if verbose_logging:
                    print(f"DEBUG: Detected directory path. Using entire path as directory: {output_directory_part}")
                    print(f"DEBUG: Keeping original filename_core: {filename_core}")
            else:
                # This is a file path (directory/filename_core)
                if user_basename:
                    filename_core = user_basename
                output_directory_part = user_dirname
                if verbose_logging:
                    print(f"DEBUG: Detected file path. filename_core set to: {filename_core}")
                    print(f"DEBUG: output_directory_part set to: {output_directory_part}")

            if is_absolute_path:
                if verbose_logging:
                    print("DEBUG: Absolute path mode.")
                if not user_dirname and user_basename:
                    status_messages.append(f"❌ Error: Absolute path template '{path_template_str}' must include an absolute directory, not just a filename.")
                    if verbose_logging:
                        print(f"DEBUG: Error - Absolute template '{path_template_str}' lacks directory part.")
                    if src_model_path_for_convert and os.path.exists(src_model_path_for_convert): os.remove(src_model_path_for_convert)
                    return ("\n".join(status_messages), "")

                if not os.path.isabs(output_directory_part):
                    status_messages.append(f"❌ Error: The directory part '{output_directory_part}' from template '{path_template_str}' is not an absolute path, but 'is_absolute_path' is True.")
                    if verbose_logging:
                        print(f"DEBUG: Error - Dir part '{output_directory_part}' is not absolute.")
                    if src_model_path_for_convert and os.path.exists(src_model_path_for_convert): os.remove(src_model_path_for_convert)
                    return ("\n".join(status_messages), "")
                final_output_directory = output_directory_part
                if verbose_logging:
                    print(f"DEBUG: Absolute mode. Final output directory: {final_output_directory}")
            else:
                if verbose_logging:
                    print("DEBUG: Relative path mode.")
                if os.path.isabs(output_directory_part):
                    abs_part_warning = f"⚠️ Warning: Path template '{path_template_str}' has an absolute directory part ('{output_directory_part}') in relative mode. This absolute part will be used directly under ComfyUI's output directory, e.g., 'ComfyUI/output{output_directory_part.lstrip(os.path.sep)}'."
                    status_messages.append(abs_part_warning)
                    if verbose_logging:
                        print(f"DEBUG: {abs_part_warning}")
                    final_output_directory = os.path.join(folder_paths.get_output_directory(), output_directory_part.lstrip(os.path.sep))
                else:
                    final_output_directory = os.path.join(folder_paths.get_output_directory(), output_directory_part)
                if verbose_logging:
                    print(f"DEBUG: Relative mode. Final output directory: {final_output_directory}")

        try:
            if verbose_logging:
                print(f"DEBUG: Attempting to create final output directory: {final_output_directory}")
            os.makedirs(final_output_directory, exist_ok=True)
            status_messages.append(f"Output directory set to: {final_output_directory}")
            if verbose_logging:
                print(f"DEBUG: Successfully ensured final output directory exists.")
        except Exception as e_mkdir:
            status_messages.append(f"❌ Error creating output directory '{final_output_directory}': {e_mkdir}")
            if verbose_logging:
                print(f"DEBUG: Exception creating output directory: {e_mkdir}")
            if src_model_path_for_convert and os.path.exists(src_model_path_for_convert): os.remove(src_model_path_for_convert)
            return ("\n".join(status_messages), "")

        # --- GGUF Conversion and Quantization ---
        final_return_path = ""
        gguf_scripts_dir = os.path.join(base_node_dir, "gguf")
        if verbose_logging:
            print(f"DEBUG: gguf_scripts_dir for 5D fix: {gguf_scripts_dir}")

        try:
            if verbose_logging:
                print("DEBUG: Entering main GGUF processing block (with tempfile.TemporaryDirectory).")
            with tempfile.TemporaryDirectory(prefix="gguf_convert_temp_") as temp_dir_for_convert_outputs:
                status_messages.append(f"Using temporary directory for GGUF conversion: {temp_dir_for_convert_outputs}")
                if verbose_logging:
                    print(f"DEBUG: temp_dir_for_convert_outputs: {temp_dir_for_convert_outputs}")

                if verbose_logging:
                    print(f"DEBUG: Calling quantizer.convert_model_to_initial_gguf with src: {src_model_path_for_convert}, temp_dir: {temp_dir_for_convert_outputs}")
                initial_gguf_path_in_temp, model_arch = quantizer.convert_model_to_initial_gguf(src_model_path_for_convert, temp_dir_for_convert_outputs)
                # quantizer.convert_model_to_initial_gguf has its own DEBUG prints
                if not initial_gguf_path_in_temp:
                    status_messages.append("❌ Error: Failed to convert model to initial GGUF (F16/BF16). Check console for convert.py script errors.")
                    if verbose_logging:
                        print("DEBUG: quantizer.convert_model_to_initial_gguf failed (returned None).")
                    raise ValueError("Initial GGUF conversion failed (convert.py error)")

                status_messages.append(f"✅ Initial GGUF created in temp: {os.path.basename(initial_gguf_path_in_temp)}")
                if model_arch: status_messages.append(f"Detected model architecture: {model_arch}")
                else: status_messages.append("⚠️ Warning: Model architecture unknown. 5D tensor fix might be skipped.")
                if verbose_logging:
                    print(f"DEBUG: Initial GGUF: {initial_gguf_path_in_temp}, Arch: {model_arch}")


                quant_types_to_process = []
                process_all_mode = quantization_type.upper() == "ALL"
                if process_all_mode:
                    quant_types_to_process = self.QUANT_TYPES
                    final_return_path = final_output_directory
                    status_messages.append(f"Processing ALL {len(quant_types_to_process)} quantization types: {', '.join(quant_types_to_process)}")
                    if verbose_logging:
                        print(f"DEBUG: 'ALL' mode selected. Processing types: {quant_types_to_process}. final_return_path set to dir: {final_return_path}")
                else:
                    quant_types_to_process = [quantization_type]
                    if verbose_logging:
                        print(f"DEBUG: Single mode selected. Processing type: {quantization_type}")

                successful_outputs_count = 0

                for idx, q_type in enumerate(quant_types_to_process):
                    q_type_upper = q_type.upper()
                    current_loop_status = [f"\n--- Processing type: {q_type_upper} ({idx+1}/{len(quant_types_to_process)}) ---"]
                    if verbose_logging:
                        print(f"DEBUG: Loop {idx+1}/{len(quant_types_to_process)} - Processing type: {q_type_upper}")

                    current_q_final_gguf_name = f"{filename_core}_{q_type_upper}.gguf"
                    current_q_final_gguf_path = os.path.join(final_output_directory, current_q_final_gguf_name)
                    if verbose_logging:
                        print(f"DEBUG: Target output path for this type: {current_q_final_gguf_path}")

                    if verbose_logging:
                        print(f"DEBUG: Calling quantizer.quantize_gguf for {q_type_upper}. Input: {initial_gguf_path_in_temp}, Output: {current_q_final_gguf_path}")
                    processed_gguf_path = quantizer.quantize_gguf(initial_gguf_path_in_temp, q_type_upper, current_q_final_gguf_path)
                    # quantizer.quantize_gguf has its own DEBUG prints

                    if not processed_gguf_path:
                        current_loop_status.append(f"❌ Error: Failed to process/quantize to {q_type_upper}.")
                        status_messages.extend(current_loop_status)
                        if verbose_logging:
                            print(f"DEBUG: quantizer.quantize_gguf failed for {q_type_upper}. Skipping this type.")
                        continue

                    current_loop_status.append(f"✅ Model processed to {q_type_upper}: {os.path.basename(processed_gguf_path)}")
                    if verbose_logging:
                        print(f"DEBUG: Successfully processed to {q_type_upper}. Path: {processed_gguf_path}")

                    if model_arch and processed_gguf_path:
                        if verbose_logging:
                            print(f"DEBUG: Model arch '{model_arch}' known. Calling quantizer.apply_5d_fix_if_needed for {processed_gguf_path}")
                        fixed_path_after_5d = quantizer.apply_5d_fix_if_needed(processed_gguf_path, model_arch, gguf_scripts_dir)
                        # quantizer.apply_5d_fix_if_needed has its own DEBUG prints
                        if fixed_path_after_5d is None:
                             current_loop_status.append(f"❌ Error during 5D tensor fix for {q_type_upper}. File '{os.path.basename(processed_gguf_path)}' might be corrupted.")
                             if verbose_logging:
                                 print(f"DEBUG: 5D fix failed for {q_type_upper}.")
                        elif fixed_path_after_5d == processed_gguf_path:
                             current_loop_status.append(f"✅ 5D tensor fix check/apply complete for {q_type_upper}.")
                             if verbose_logging:
                                 print(f"DEBUG: 5D fix check/apply complete for {q_type_upper}.")
                             successful_outputs_count +=1
                             if not process_all_mode: final_return_path = processed_gguf_path
                    elif not model_arch:
                        current_loop_status.append(f"ℹ️ Skipping 5D tensor fix for {q_type_upper} (model architecture unknown).")
                        if verbose_logging:
                            print(f"DEBUG: Skipping 5D fix for {q_type_upper} (no model_arch).")
                        successful_outputs_count +=1
                        if not process_all_mode: final_return_path = processed_gguf_path
                    else: # This case should ideally not be reached if processed_gguf_path was None and continue was hit.
                        if verbose_logging:
                            print(f"DEBUG: Fallthrough case after 5D fix logic for {q_type_upper} (processed_gguf_path might be None or arch unknown). This indicates an issue if processed_gguf_path was valid.")
                        if processed_gguf_path : # If quantize was successful but arch unknown for fix
                            successful_outputs_count +=1
                            if not process_all_mode: final_return_path = processed_gguf_path


                    status_messages.extend(current_loop_status)

                if successful_outputs_count == 0:
                    if verbose_logging:
                        print("DEBUG: No GGUF files were successfully created or processed in the loop.")
                    raise ValueError("No GGUF files were successfully created or processed during quantization loop.")

                status_messages.append(f"\n🎉 Successfully processed. {successful_outputs_count} GGUF file(s) created/updated in '{final_output_directory}'.")
                if verbose_logging:
                    print(f"DEBUG: Loop finished. Successful outputs: {successful_outputs_count}.")

            if verbose_logging:
                print("DEBUG: Exited GGUF processing block (tempfile.TemporaryDirectory scope ended).")

        except Exception as e:
            if verbose_logging:
                print(f"DEBUG: Exception during main GGUF processing block: {e}")
            status_messages.append(f"\n❌ An critical error occurred during GGUF processing: {e}")
            import traceback
            tb_str = traceback.format_exc()
            if verbose_logging:
                print(f"DEBUG: Traceback for GGUF processing exception: {tb_str}")
            status_messages.append(f"Traceback: {tb_str}")
            final_return_path = ""
        finally:
            if verbose_logging:
                print("DEBUG: Entering final cleanup block (finally).")
            if temp_model_input_path and os.path.exists(temp_model_input_path):
                try:
                    if verbose_logging:
                        print(f"DEBUG: Removing temporary input UNET: {temp_model_input_path}")
                    os.remove(temp_model_input_path)
                    status_messages.append(f"🗑️ Cleaned up temporary input UNET: {os.path.basename(temp_model_input_path)}")
                    if verbose_logging:
                        print(f"DEBUG: Successfully removed {temp_model_input_path}")
                except Exception as e_rem:
                    status_messages.append(f"⚠️ Warning: Failed to clean temporary UNET file '{temp_model_input_path}': {e_rem}")
                    if verbose_logging:
                        print(f"DEBUG: Failed to remove {temp_model_input_path}: {e_rem}")
            else:
                if verbose_logging:
                    print(f"DEBUG: No temporary input UNET file to remove (Path: {temp_model_input_path}, Exists: {os.path.exists(temp_model_input_path) if temp_model_input_path else 'N/A'})")

        if not final_return_path:
            status_messages.append(f"\n❌ Processing failed. No valid output path determined. Check logs.")
            if verbose_logging:
                print("DEBUG: final_return_path is empty at the end. Processing failed.")
            return ("\n".join(status_messages), "")

        if not process_all_mode and not os.path.exists(final_return_path):
            status_messages.append(f"\n❌ Error: Final GGUF file '{final_return_path}' not found after processing.")
            if verbose_logging:
                print(f"DEBUG: Single mode, but final_return_path '{final_return_path}' does not exist.")
            return ("\n".join(status_messages), "")

        if verbose_logging:
            print(f"DEBUG: Returning from quantize_diffusion_model. Status messages collected. Final return path: {final_return_path}")
        return ("\n".join(status_messages), final_return_path)


# ComfyUI Registration is handled by __init__.py
