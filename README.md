# VieNeu-TTS

[![Github](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

[<img width="600" height="595" alt="image" src="https://github.com/user-attachments/assets/66c098c4-d184-4e7a-826a-ba8c6c556fab" />](https://github.com/user-attachments/assets/5ad53bc9-e816-41a7-9474-ea470b1cbfdd)

**VieNeu-TTS** là mô hình Text-to-Speech (TTS) tiếng Việt đầu tiên chạy trên thiết bị cá nhân với khả năng nhân bản giọng nói tức thì. Được fine-tune từ [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air), VieNeu-TTS mang đến giọng nói tiếng Việt tự nhiên, siêu chân thực với hiệu suất thời gian thực trên CPU.

Dựa trên backbone Qwen 0.5B LLM, VieNeu-TTS kết hợp giữa tốc độ, kích thước nhỏ gọn và chất lượng âm thanh cao - hoàn hảo cho các ứng dụng voice agent, trợ lý ảo, đồ chơi tương tác và các ứng dụng yêu cầu bảo mật cao chạy trên thiết bị local.

**Tác giả**: Phạm Nguyễn Ngọc Bảo

## ✨ Tính năng

- 🎙️ **Tổng hợp giọng nói tiếng Việt tự nhiên** với chất lượng cao (24kHz)
- 🚀 **Voice Cloning tức thì** - chỉ cần một đoạn audio mẫu ngắn
- 💻 **Chạy trên thiết bị local** - không cần kết nối internet để inference
- 🎯 **Đa dạng giọng nói** - hỗ trợ nhiều giọng nam/nữ miền Nam
- ⚡ **Hiệu suất cao** - có thể chạy realtime trên CPU/GPU
- 🔧 **Dễ tích hợp** - API đơn giản, hỗ trợ Gradio web interface

## Chi tiết mô hình

VieNeu-TTS được xây dựng dựa trên kiến trúc NeuTTS Air với các thành phần chính:

- **Base Model**: Qwen 0.5B
- **Audio Codec**: NeuCodec
- **Format**: Safetensors và GGUF (Q8, Q4) cho suy luận hiệu quả trên thiết bị
- **Trách nhiệm**: Audio đầu ra có watermark tích hợp
- **Tốc độ suy luận**: Sinh giọng nói thời gian thực trên thiết bị tầm trung
- **Tiêu thụ điện năng**: Tối ưu cho thiết bị di động và nhúng
- **Dataset huấn luyện**: 
  - [VieNeuCodec-dataset](https://huggingface.co/datasets/pnnbao-ump/VieNeuCodec-dataset) - 74.9k mẫu audio tiếng Việt
  - Fine-tuned từ base model đã được train trên [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)
 
## Bắt đầu

### Clone Git Repo

```bash
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS
```

### Cài đặt espeak (dependency bắt buộc)

Tham khảo hướng dẫn chi tiết tại: https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md
```bash
# Mac OS
brew install espeak

# Ubuntu/Debian
sudo apt install espeak

# Arch Linux
paru -S aur/espeak

# Windows
# Tải và cài đặt từ: https://github.com/espeak-ng/espeak-ng/releases
# Mặc định cài vào: C:\Program Files\eSpeak NG\
# Code sẽ tự động nhận diện đường dẫn này
```

**Lưu ý cho macOS:**
- Nếu vẫn gặp lỗi sau khi cài đặt, hãy đảm bảo đã set đúng environment variables
- Kiểm tra cài đặt bằng lệnh: `echo 'test' | espeak-ng -x -q --ipa -v en-us`
- Nếu output hiển thị phiên âm IPA, espeak đã được cài đặt thành công

### Cài đặt Python dependencies

File requirements bao gồm các dependencies cần thiết để chạy model với PyTorch. Khi sử dụng ONNX decoder hoặc GGML model, một số dependencies (như PyTorch) có thể không cần thiết.

Inference tương thích và đã được test trên python>=3.11.

```bash
# Cài đặt từ requirements.txt
pip install -r requirements.txt

# Hoặc sử dụng uv (nếu có pyproject.toml)
uv pip install -r requirements.txt

# Hoặc cài đặt từ pyproject.toml
pip install -e .
```

**Lưu ý**: Nếu bạn sử dụng GPU, hãy đảm bảo cài đặt PyTorch với hỗ trợ CUDA phù hợp:
```bash
# Xem hướng dẫn tại: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```
## 📁 Cấu trúc dự án

```
VieNeuTTS/
├── vieneutts.py              # Module chính chứa class VieNeuTTS
├── main.py                   # Script ví dụ sử dụng cơ bản
├── gradio_app.py             # Ứng dụng Gradio để chạy web demo (local)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project configuration (nếu dùng uv)
├── README.md                 # File này
├── sample/                   # Thư mục chứa các file audio và text mẫu
│   ├── id_0001.wav/txt      # Nam 1 - miền Nam
│   ├── id_0002.wav/txt      # Nữ 1 - miền Nam
│   ├── id_0003.wav/txt      # Nam 2 - miền Nam
│   ├── id_0004.wav/txt      # Nữ 2 - miền Nam
│   ├── id_0005.wav/txt      # Nam 3 - miền Nam
│   └── id_0007.wav/txt      # Nam 4 - miền Nam
├── VieNeuTTS/                # Thư mục con (cho Hugging Face Spaces)
│   ├── app.py               # Gradio app cho Spaces
│   └── vieneutts.py         # Module VieNeuTTS (bản sao)
└── output_audio/             # Thư mục chứa kết quả (tự tạo khi chạy)
```

## 💻 Cách sử dụng

### 1. Sử dụng qua Python API

## Ví dụ cơ bản

```python
from vieneutts import VieNeuTTS
import soundfile as sf
import os

# Văn bản cần tổng hợp
input_texts = [
    "Các khóa học trực tuyến đang giúp học sinh tiếp cận kiến thức mọi lúc mọi nơi.",
    "Các nghiên cứu về bệnh Alzheimer cho thấy tác dụng tích cực của các bài tập trí não.",
    "Một tiểu thuyết trinh thám hiện đại dẫn dắt độc giả qua những tình tiết phức tạp, bí ẩn.",
]

output_dir = "./output_audio"
os.makedirs(output_dir, exist_ok=True)

# Đường dẫn file tham chiếu
# Nam miền Nam
ref_audio_path = "./sample/id_0001.wav"
ref_text_path = "./sample/id_0001.txt"
# Nữ miền Nam
# ref_audio_path = "./sample/id_0002.wav"
# ref_text_path = "./sample/id_0002.txt"

ref_text = open(ref_text_path, "r", encoding="utf-8").read()

# Khởi tạo model
tts = VieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    backbone_device="cuda",  # hoặc "cpu" nếu không có GPU
    codec_repo="neuphonic/neucodec",
    codec_device="cuda"  # hoặc "cpu" nếu không có GPU
)

print("Encoding reference audio")
ref_codes = tts.encode_reference(ref_audio_path)

# Tổng hợp giọng nói cho nhiều văn bản
for i, text in enumerate(input_texts, 1):
    print(f"Generating audio for example {i}: {text}")
    wav = tts.infer(text, ref_codes, ref_text)
    output_path = os.path.join(output_dir, f"output_{i}.wav")
    sf.write(output_path, wav, 24000)
    print(f"Saved to {output_path}")
```

### 2. Sử dụng qua Gradio Web Interface (Local)

Chạy ứng dụng web đơn giản với giao diện trực quan:

```bash
python gradio_app.py
```

Sau đó mở trình duyệt và truy cập `http://127.0.0.1:7860`

**Tính năng của Gradio App:**
- ✅ Chọn giọng từ 6 giọng mẫu có sẵn
- ✅ Upload audio tùy chỉnh để clone giọng
- ✅ Preview và download kết quả
- ✅ Có ví dụ mẫu sẵn để thử nghiệm

### 3. Sử dụng script main.py

Script `main.py` cung cấp ví dụ tổng hợp nhiều văn bản cùng lúc:

```bash
python main.py
```

Kết quả sẽ được lưu trong thư mục `output_audio/`.

**Lưu ý**: Bạn có thể chỉnh sửa trong `main.py`:
- Chọn giọng mẫu (id_0001 đến id_0007)
- Thay đổi văn bản đầu vào
- Tùy chỉnh device (cuda/cpu)

### 4. Giọng mẫu có sẵn

Trong thư mục `sample/`, có 6 giọng mẫu sẵn có:

| File | Giới tính | Miền | Mô tả |
|------|-----------|------|-------|
| `id_0001` | Nam | Miền Nam | Giọng nam 1 |
| `id_0002` | Nữ | Miền Nam | Giọng nữ 1 |
| `id_0003` | Nam | Miền Nam | Giọng nam 2 |
| `id_0004` | Nữ | Miền Nam | Giọng nữ 2 |
| `id_0005` | Nam | Miền Nam | Giọng nam 3 |
| `id_0007` | Nam | Miền Nam | Giọng nam 4 |

**Quy ước**: 
- File số **lẻ** (1, 3, 5, 7) → Giọng **Nam**
- File số **chẵn** (2, 4) → Giọng **Nữ**

## ⚠️ Khuyến cáo

Vui lòng không sử dụng mô hình này cho mục đích xấu hoặc vi phạm pháp luật, bao gồm:

- Mạo danh giọng nói người khác mà không có sự đồng ý
- Tạo nội dung sai sự thật, lừa đảo
- Vi phạm quyền riêng tư hoặc quyền sở hữu trí tuệ
- Các hành vi vi phạm pháp luật khác

Hãy tôn trọng quyền riêng tư và quyền sở hữu trí tuệ của người khác.

## ⚠️ Giới hạn

- Mô hình có thể không phát âm chính xác 100% các từ tiếng Việt phức tạp hoặc từ vựng chuyên ngành
- Chất lượng đầu ra phụ thuộc nhiều vào chất lượng của audio tham chiếu
- Hiệu suất có thể giảm với văn bản quá dài (khuyến nghị chia nhỏ văn bản dài, tối đa ~500 ký tự)
- Văn bản đầu vào nên ở dạng chuẩn, tránh viết tắt hoặc ký tự đặc biệt không chuẩn

## 🐛 Xử lý lỗi thường gặp

### Lỗi: "Failed to import espeak"

**Nguyên nhân**: Chưa cài đặt hoặc chưa cấu hình đúng eSpeak NG

**Giải pháp**:
- **Windows**: Đảm bảo đã cài đặt eSpeak NG vào `C:\Program Files\eSpeak NG\`
- **Linux**: Chạy `sudo apt install espeak` hoặc `sudo apt install espeak-ng`
- **MacOS**: Chạy `brew install espeak` hoặc `brew install espeak-ng`

### Lỗi: "CUDA out of memory"

**Nguyên nhân**: GPU không đủ bộ nhớ

**Giải pháp**:
- Sử dụng CPU: đổi `backbone_device="cpu"` và `codec_device="cpu"`
- Hoặc sử dụng model quantized (GGUF Q4/Q8)

### Lỗi: "No valid speech tokens found"

**Nguyên nhân**: Model không generate được speech tokens hợp lệ

**Giải pháp**:
- Kiểm tra lại text input (không để trống, không quá dài)
- Kiểm tra audio reference (đảm bảo file hợp lệ)
- Thử với text ngắn hơn

## 📚 Tài liệu tham khảo

- [GitHub Repository](https://github.com/pnnbao97/VieNeu-TTS)
- [Hugging Face Model](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
- [NeuTTS Air Base Model](https://huggingface.co/neuphonic/neutts-air)
- [Hướng dẫn Finetune](https://github.com/pnnbao-ump/VieNeuTTS/blob/main/finetune.ipynb)
- [Dataset huấn luyện](https://huggingface.co/datasets/pnnbao-ump/VieNeuCodec-dataset)

## 📄 License

Apache 2.0

## Trích dẫn

Nếu bạn sử dụng VieNeu-TTS trong nghiên cứu hoặc ứng dụng của mình, vui lòng trích dẫn:

```bibtex
@misc{vieneutts2025,
  title={VieNeu-TTS: Vietnamese Text-to-Speech with Instant Voice Cloning},
  author={Pham Nguyen Ngoc Bao},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/pnnbao-ump/VieNeu-TTS}}
}
```

Và base model NeuTTS Air:

```bibtex
@misc{neuttsair2025,
  title={NeuTTS Air: On-Device Speech Language Model with Instant Voice Cloning},
  author={Neuphonic},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/neuphonic/neutts-air}}
}
```

## Liên hệ

- **GitHub**: [pnnbao97](https://github.com/pnnbao97)
- **Hugging Face**: [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Facebook**: [Phạm Nguyễn Ngọc Bảo](https://www.facebook.com/bao.phamnguyenngoc.5)

## Ghi nhận

Dự án này được xây dựng dựa trên [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air) của Neuphonic. Xin gửi lời cảm ơn chân thành đến đội ngũ Neuphonic đã tạo ra mô hình base xuất sắc này và công khai cho cộng đồng.

---

**Lưu ý**: Đây là phiên bản nghiên cứu và thử nghiệm. Vui lòng báo cáo các vấn đề hoặc đóng góp cải tiến qua [GitHub Issues](https://github.com/pnnbao97/VieNeu-TTS/issues).

---

## 🙏 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork the repository
2. Tạo branch mới cho feature của bạn (`git checkout -b feature/AmazingFeature`)
3. Commit các thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📞 Hỗ trợ

Nếu bạn gặp vấn đề hoặc có câu hỏi:
- Tạo issue trên [GitHub](https://github.com/pnnbao97/VieNeu-TTS/issues)
- Liên hệ qua [Facebook](https://www.facebook.com/bao.phamnguyenngoc.5)

---

**Made with ❤️ for Vietnamese TTS community**








