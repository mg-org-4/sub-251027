import requests
import json
import os
import time


class RH_VideoUploaderPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apiConfig": ("STRUCT",),
                "video_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "upload_and_get_filename"
    CATEGORY = "RunningHub"
    OUTPUT_NODE = False

    def upload_and_get_filename(self, apiConfig, video_path):
        if not isinstance(apiConfig, dict) or not apiConfig.get("apiKey") or not apiConfig.get("base_url"):
            raise ValueError("Invalid or missing apiConfig structure provided to RH_VideoUploaderPath.")

        if not isinstance(video_path, str) or video_path.strip() == "":
            raise ValueError("No video_path provided. Please input an absolute video path.")

        video_path = video_path.strip().strip('"').strip("'")

        if not os.path.isabs(video_path):
            raise ValueError(f"video_path must be an absolute path: {video_path}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        apiKey = apiConfig["apiKey"]
        baseUrl = apiConfig["base_url"]

        upload_api_url = f"{baseUrl}/task/openapi/upload"
        headers = {
            "User-Agent": "ComfyUI-RH_VideoUploaderPath/1.0"
        }
        data = {
            "apiKey": apiKey,
            "fileType": "video"
        }

        print(f"RH_VideoUploaderPath: Uploading {video_path} to {upload_api_url}...")

        max_retries = 5
        retry_delay = 1
        last_exception = None
        response = None

        for attempt in range(max_retries):
            try:
                with open(video_path, "rb") as f:
                    files = {
                        "file": (os.path.basename(video_path), f)
                    }
                    response = requests.post(upload_api_url, headers=headers, data=data, files=files)
                    print(
                        f"RH_VideoUploaderPath: Upload attempt {attempt + 1}/{max_retries} - Status Code: {response.status_code}"
                    )
                    response.raise_for_status()

                break

            except requests.exceptions.Timeout as e:
                last_exception = e
                print(f"RH_VideoUploaderPath: Upload attempt {attempt + 1} timed out.")
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                print(f"RH_VideoUploaderPath: Upload attempt {attempt + 1} connection error: {e}")
            except requests.exceptions.RequestException as e:
                last_exception = e
                print(f"RH_VideoUploaderPath: Upload attempt {attempt + 1} failed: {e}")
                if e.response is not None:
                    print(f"RH_VideoUploaderPath: Response content on error: {e.response.text}")

            if attempt < max_retries - 1:
                print(f"RH_VideoUploaderPath: Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"RH_VideoUploaderPath: Max retries ({max_retries}) reached.")
                raise ConnectionError(
                    f"Failed to upload video to RunningHub API after {max_retries} attempts. Last error: {last_exception}"
                ) from last_exception

        if response is None:
            raise ConnectionError(
                f"Upload failed after {max_retries} attempts, no response received. Last error: {last_exception}"
            )

        try:
            response_json = response.json()
            print(f"RH_VideoUploaderPath: Upload API Response JSON: {response_json}")
        except json.JSONDecodeError as e:
            print(f"RH_VideoUploaderPath: Failed to decode JSON response: {response.text}")
            raise ValueError(f"Failed to decode API response after successful upload: {e}") from e

        if response_json.get("code") != 0:
            raise ValueError(
                f"RunningHub API reported an error after upload: {response_json.get('msg', 'Unknown API error')}"
            )

        rh_data = response_json.get("data", {})
        uploaded_filename = None
        if isinstance(rh_data, dict):
            uploaded_filename = rh_data.get("fileName")
        elif isinstance(rh_data, str):
            uploaded_filename = rh_data

        if not isinstance(uploaded_filename, str) or not uploaded_filename:
            raise ValueError(
                "Upload succeeded but 'fileName' (or compatible field) not found in RunningHub API response.data."
            )

        print(f"RH_VideoUploaderPath: Upload successful. RunningHub filename/ID: {uploaded_filename}")
        return (uploaded_filename,)
