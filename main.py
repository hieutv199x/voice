from vieneutts import VieNeuTTS
import soundfile as sf
import os

input_texts = [
    "Các khóa học trực tuyến đang giúp học sinh tiếp cận kiến thức mọi lúc mọi nơi. Giáo viên sử dụng video, bài tập tương tác và thảo luận trực tuyến để nâng cao hiệu quả học tập.",

    "Các nghiên cứu về bệnh Alzheimer cho thấy tác dụng tích cực của các bài tập trí não và chế độ dinh dưỡng lành mạnh, giúp giảm tốc độ suy giảm trí nhớ ở người cao tuổi.",

    "Một tiểu thuyết trinh thám hiện đại dẫn dắt độc giả qua những tình tiết phức tạp, bí ẩn, kết hợp yếu tố tâm lý sâu sắc khiến người đọc luôn hồi hộp theo dõi diễn biến câu chuyện.",

    "Các nhà khoa học nghiên cứu gen người phát hiện những đột biến mới liên quan đến bệnh di truyền. Điều này giúp nâng cao khả năng chẩn đoán và điều trị.",
]

output_dir = "./output_audio"
os.makedirs(output_dir, exist_ok=True)

def main(backbone="pnnbao-ump/VieNeu-TTS", codec="neuphonic/neucodec"):
    """
    Trong thư mục sample, có 7 file wav và 7 file txt, các file wav và txt có cùng tên. Đây là những file chuẩn được mình chuẩn bị cho các bạn test.
    Ví dụ: id_0001.wav và id_0001.txt
    Ví dụ: id_0002.wav và id_0002.txt
    Ví dụ: id_0003.wav và id_0003.txt
    Ví dụ: id_0004.wav và id_0004.txt
    Ví dụ: id_0005.wav và id_0005.txt
    Ví dụ: id_0006.wav và id_0006.txt
    Ví dụ: id_0007.wav và id_0007.txt
    Các file số lẻ là nam giới, các file số chẵn là nữ giới. Các bạn có thể chọn file wav và txt tương ứng để test.
    Lưu ý: model vẫn có thể clone giọng của audio bạn đưa vào (kèm text tương ứng). Tuy nhiên, chất lượng có thể không được tốt như các file trong thư mục sample. Các bạn có thể finetune model này trên giọng các bạn cần clone để có chất lượng tốt nhất.
    Các bạn có thể tham khảo cách finetune model tại https://github.com/pnnbao-ump/VieNeuTTS/blob/main/finetune.ipynb
    """
    # Nam miền Nam
    ref_audio_path = "./sample/id_0004.wav"
    ref_text = "./sample/id_0004.txt"
    # Nữ miền Nam
    # ref_audio_path = "./sample/id_0002.wav"
    # ref_text = "./sample/id_0002.txt"

    ref_text = open(ref_text, "r", encoding="utf-8").read()
    if not ref_audio_path or not ref_text:
        print("No reference audio or text provided.")
        return None

    # Initialize VieNeuTTS with the desired model and codec
    tts = VieNeuTTS(
        backbone_repo=backbone,
        backbone_device="cuda",
        codec_repo=codec,
        codec_device="cuda"
    )

    print("Encoding reference audio")
    ref_codes = tts.encode_reference(ref_audio_path)

    # Loop through all input texts
    for i, text in enumerate(input_texts, 1):
        print(f"Generating audio for example {i}: {text}")
        wav = tts.infer(text, ref_codes, ref_text)
        output_path = os.path.join(output_dir, f"output_{i}.wav")
        sf.write(output_path, wav, 24000)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()