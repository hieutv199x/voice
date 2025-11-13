import gradio as gr
import soundfile as sf
import tempfile
import torch

print("⏳ Đang khởi động VieNeu-TTS...")
# Import vieneutts
from vieneutts import VieNeuTTS

# Khởi tạo model
print("📦 Đang tải model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Sử dụng thiết bị: {device.upper()}")

tts = VieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    backbone_device=device,
    codec_repo="neuphonic/neucodec",
    codec_device=device
)
print("✅ Model đã tải xong!")

# Danh sách giọng mẫu (bỏ id_0006)
VOICE_SAMPLES = {
    "Nam 1 (id_0001)": {
        "audio": "./sample/id_0001.wav",
        "text": "./sample/id_0001.txt"
    },
    "Nữ 1 (id_0002)": {
        "audio": "./sample/id_0002.wav",
        "text": "./sample/id_0002.txt"
    },
    "Nam 2 (id_0003)": {
        "audio": "./sample/id_0003.wav",
        "text": "./sample/id_0003.txt"
    },
    "Nữ 2 (id_0004)": {
        "audio": "./sample/id_0004.wav",
        "text": "./sample/id_0004.txt"
    },
    "Nam 3 (id_0005)": {
        "audio": "./sample/id_0005.wav",
        "text": "./sample/id_0005.txt"
    },
    "Nam 4 (id_0007)": {
        "audio": "./sample/id_0007.wav",
        "text": "./sample/id_0007.txt"
    }
}

def synthesize_speech(text, voice_choice, custom_audio=None, custom_text=None):
    """
    Tổng hợp giọng nói từ văn bản
    """
    try:
        # Kiểm tra text input
        if not text or text.strip() == "":
            return None, "❌ Vui lòng nhập văn bản cần tổng hợp"
        
        # Giới hạn độ dài text
        if len(text) > 250:
            return None, "❌ Văn bản quá dài! Vui lòng nhập tối đa 250 ký tự"
        
        # Xác định reference audio và text
        if custom_audio is not None and custom_text:
            ref_audio_path = custom_audio
            ref_text = custom_text
            print("🎨 Sử dụng giọng tùy chỉnh")
        elif voice_choice in VOICE_SAMPLES:
            ref_audio_path = VOICE_SAMPLES[voice_choice]["audio"]
            ref_text_path = VOICE_SAMPLES[voice_choice]["text"]
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text = f.read()
            print(f"🎤 Sử dụng giọng: {voice_choice}")
        else:
            return None, "❌ Vui lòng chọn giọng hoặc tải lên audio tùy chỉnh"
        
        # Encode reference audio
        print(f"📝 Đang xử lý: {text[:50]}...")
        ref_codes = tts.encode_reference(ref_audio_path)
        
        # Tổng hợp giọng nói
        print(f"🎵 Đang tổng hợp giọng nói trên {device.upper()}...")
        wav = tts.infer(text, ref_codes, ref_text)
        
        # Lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, wav, 24000)
            output_path = tmp_file.name
        
        print("✅ Hoàn thành!")
        return output_path, f"✅ Tổng hợp thành công trên {device.upper()}!"
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Lỗi: {str(e)}"

# Các ví dụ mẫu
examples = [
    ["Legacy là một bộ phim đột phá về mặt âm nhạc, quay phim, hiệu ứng đặc biệt, và tôi rất mừng vì cuối cùng nó cũng được cả giới phê bình lẫn người hâm mộ đánh giá lại. Chúng ta đã quá bất công với bộ phim này vào năm 2010.", "Nam 1 (id_0001)"],
    ["Từ nhiều nguồn tài liệu lịch sử, có thể thấy nuôi con theo phong cách Do Thái không chỉ tốt cho đứa trẻ mà còn tốt cho cả các bậc cha mẹ.", "Nữ 1 (id_0002)"],
    ["Các bác sĩ đang nghiên cứu một loại vaccine mới chống lại virus cúm mùa. Thí nghiệm lâm sàng cho thấy phản ứng miễn dịch mạnh mẽ và ít tác dụng phụ, mở ra hy vọng phòng chống dịch bệnh hiệu quả hơn trong tương lai.", "Nam 2 (id_0003)"],
]

# Custom CSS
custom_css = """
.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}
.contain {
    max-width: 1000px !important;
    margin: 0 auto !important;
}
#warning {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
#info {
    background-color: #d1ecf1;
    border: 1px solid #17a2b8;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
"""

# Tạo giao diện Gradio
with gr.Blocks(title="VieNeu-TTS Local", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ VieNeu-TTS: Vietnamese Text-to-Speech (Local Version)

    Hệ thống tổng hợp tiếng nói tiếng Việt được **finetune từ NeuTTS-Air** - một mô hình TTS tiên tiến sử dụng Large Language Model và Neural Codec.

    Tác giả: [Phạm Nguyễn Ngọc Bảo](https://github.com/pnnbao97)  
    Model: [VieNeu-TTS](https://huggingface.co/pnnbao-ump/VieNeu-TTS)  
    Code: [GitHub](https://github.com/pnnbao97/VieNeu-TTS)
    """)
    
    with gr.Row():
        with gr.Column():
            # Input text
            text_input = gr.Textbox(
                label="📝 Văn bản đầu vào (tối đa 250 ký tự)",
                placeholder="Nhập văn bản tiếng Việt...",
                lines=4,
                max_lines=6,
                value="Legacy là một bộ phim đột phá về mặt âm nhạc, quay phim, hiệu ứng đặc biệt, và tôi rất mừng vì cuối cùng nó cũng được cả giới phê bình lẫn người hâm mộ đánh giá lại. Chúng ta đã quá bất công với bộ phim này vào năm 2010."
            )
            
            # Character counter
            char_count = gr.Markdown("209 / 250 ký tự")
            
            # Voice selection
            voice_select = gr.Radio(
                choices=list(VOICE_SAMPLES.keys()),
                label="🎤 Chọn giọng mẫu",
                value="Nam 1 (id_0001)",
                info="Giọng lẻ: Nam | Giọng chẵn: Nữ"
            )
            
            # Custom voice option
            with gr.Accordion("🎨 Hoặc sử dụng giọng tùy chỉnh", open=False):
                gr.Markdown("""
                **Hướng dẫn:**
                - Upload file audio (.wav) và nhập nội dung text chính xác tương ứng
                - **Lưu ý:** Chất lượng có thể không tốt bằng các giọng mẫu trong thư mục sample
                - Để có kết quả tốt nhất, hãy finetune model trên giọng của bạn tại: [finetune.ipynb](https://github.com/pnnbao-ump/VieNeuTTS/blob/main/finetune.ipynb)
                """)
                custom_audio = gr.Audio(
                    label="File audio mẫu",
                    type="filepath"
                )
                custom_text = gr.Textbox(
                    label="Nội dung của audio mẫu",
                    placeholder="Nhập chính xác nội dung...",
                    lines=2
                )
            
            # Submit button
            submit_btn = gr.Button("🎵 Tổng hợp giọng nói", variant="primary", size="lg")
        
        with gr.Column():
            # Output
            audio_output = gr.Audio(label="🔊 Kết quả")
            status_output = gr.Textbox(label="📊 Trạng thái", interactive=False)
    
    # Examples
    gr.Markdown("### 💡 Ví dụ nhanh")
    gr.Examples(
        examples=examples,
        inputs=[text_input, voice_select],
        outputs=[audio_output, status_output],
        fn=synthesize_speech,
        cache_examples=False
    )
    
    # Update character count
    def update_char_count(text):
        count = len(text) if text else 0
        color = "red" if count > 250 else "green"
        return f"<span style='color: {color}'>{count} / 250 ký tự</span>"
    
    text_input.change(
        fn=update_char_count,
        inputs=[text_input],
        outputs=[char_count]
    )
    
    # Event handler
    submit_btn.click(
        fn=synthesize_speech,
        inputs=[text_input, voice_select, custom_audio, custom_text],
        outputs=[audio_output, status_output]
    )
    
    gr.Markdown("""
    ---
    ### 📌 Thông tin về giọng mẫu
    
    **Giọng có sẵn trong thư mục sample:**
    - `id_0001.wav/txt` - Nam 1 ✅
    - `id_0002.wav/txt` - Nữ 1 ✅
    - `id_0003.wav/txt` - Nam 2 ✅
    - `id_0004.wav/txt` - Nữ 2 ✅
    - `id_0005.wav/txt` - Nam 3 ✅
    - `id_0007.wav/txt` - Nam 4 ✅
    
    *Các file số lẻ: Nam giới | Các file số chẵn: Nữ giới*
    
    **Liên kết:**
    - [GitHub Repository](https://github.com/pnnbao97/VieNeu-TTS)
    - [Model Card](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
    - [Hướng dẫn Finetune](https://github.com/pnnbao-ump/VieNeuTTS/blob/main/finetune.ipynb)
    
    <sub>Powered by VieNeu-TTS | Built with ❤️ for Vietnamese TTS</sub>
    """)

# Launch
if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )