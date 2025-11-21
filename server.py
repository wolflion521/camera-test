import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os
import base64
from http import HTTPStatus
from dashscope import MultiModalConversation

# ================= 配置你的 API KEY =================
os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # 填入你的 Key
# ==================================================

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_frames(video_path, num_frames=3):
    """从视频中均匀提取 3 张关键帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if total_frames == 0: return []
    
    frames_content = []
    # 比如取开头、中间、结尾
    indices = [0, total_frames // 2, total_frames - 5]
    
    for i in indices:
        if i < 0: i = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # 压缩一下图片，避免传给 API 太大
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            # 转成 base64 也可以，或者保存临时文件
            temp_img_name = f"temp_frame_{i}.jpg"
            with open(temp_img_name, "wb") as f:
                f.write(buffer)
            frames_content.append(f"file://{os.path.abspath(temp_img_name)}")
            
    cap.release()
    return frames_content

@app.post("/upload_and_analyze")
async def analyze_video(file: UploadFile = File(...)):
    print(f"收到视频: {file.filename}, 开始处理...")
    
    # 1. 保存视频到本地
    temp_video_path = "temp_video.webm"
    with open(temp_video_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # 2. 视频抽帧 (为了省钱和快，我们不传整个视频，只传关键帧)
        # Qwen-VL 支持直接传视频，但通过 API 传大文件容易超时，抽帧是最稳的
        frame_urls = extract_frames(temp_video_path)
        
        if not frame_urls:
            return {"result": "视频解析失败，未提取到画面"}

        # 3. 构造 Prompt 发送给 Qwen-VL-Max
        content_list = [{"image": url} for url in frame_urls]
        content_list.append({"text": "这是视频中的三个关键画面。请分析视频中孩子的行为。是否发生跌倒、碰撞等危险？请输出【安全】或【危险】，并简述原因。"})

        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]

        print("正在请求云端大模型...")
        response = MultiModalConversation.call(model='qwen-vl-max', messages=messages)
        
        if response.status_code == HTTPStatus.OK:
            result_text = response.output.choices[0].message.content[0]['text']
            print(f"分析完成: {result_text}")
            return {"status": "success", "result": result_text}
        else:
            return {"status": "error", "result": f"API 错误: {response.message}"}

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "result": str(e)}

if __name__ == "__main__":
    # 必须生成 SSL 证书，否则手机无法录像
    # 命令: openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")