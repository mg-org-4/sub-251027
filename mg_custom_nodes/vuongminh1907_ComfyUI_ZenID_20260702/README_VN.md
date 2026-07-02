# 🌟 ZenID: Face Swap 🌟
Repo này được lấy cảm hứng và modify từ [**InstantID**](https://github.com/instantX-research/InstantID) và [**InstantID Comfy**](https://github.com/cubiq/ComfyUI_InstantID)

**ZenID Node** đã được chỉnh sửa lại để phục vụ các tác vụ chuyên biệt như _Face Swap_

**✨ Hãy ủng hộ sự phát triển thêm của dự án bằng cách đánh dấu sao cho dự án! ✨**

## 📑 **Mục Lục** 
1. [Tính Năng ZenID](#zenid-features) 
    * [Face Swap](#zenid-face-swap) 
    * [Face Combine](#zenid-face-combine) 
2. [Cài Đặt](#installation)

## 🌟 **Tính Năng ZenID** <a name="zenid-features"></a>

### 🔗 **ZenID Face Swap** <a name="zenid-face-swap"></a>
- **Workflows**
    File mẫu [`ZenID_FaceSwap.json`](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/workflow/ZenID_FaceSwap.json) được đính kèm trong thư mục `workflow`.

    Ở bước này, bạn có thể dùng mask tự vẽ hoặc để cho nodes tự sinh ra mask để swap

    **NOTE: Dùng tay tự vẽ mask thì kết quả đẹp hơn**

    ![ZenID Face Swap Example](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenid_faceswap.png)

- **Ví Dụ**
    - **Hình Ảnh Gốc**  
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/blackwukong.png" width="300" /> 
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/domixi.png" width="300" />

    - **Kết Quả**  
       <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/result_faceswap.png" width="600" />


### 🔗 **ZenID Face Combine** <a name="zenid-face-swap"></a>
- **Workflows**

    File mẫu [`ZenID_combineFace.json`](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/workflow/ZenID_combineFace.json) được đính kèm trong thư mục `workflow`.
    ![ZenID Face Combine Example](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenid_combineface.png)
- **Ví Dụ**
    - **Hình Ảnh Gốc**  
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/haitu.jpg" width="300" /> 
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/sontung.jpg" width="300" />

    - **Kết Quả**  
       <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/result_facecombine.png" width="600" />

## ⚙️ Cài đặt <a name="installation"></a>
1. Clone repo của ComfyUI 
```
git clone https://github.com/comfyanonymous/ComfyUI
```
2. Sao chép hoặc tải về repo này vào thư mục `ComfyUI/custom_nodes/`.
```
cd ComfyUI/custom_nodes/
git clone https://github.com/vuongminh1907/ComfyUI_ZenID
```
3. Tải model để có thể chạy được
```
pip install -r ComfyUI_ZenID/requirements.txt
python ComfyUI_ZenID/downloadmodel.py
```
4. Khởi chạy ComfyUI.

