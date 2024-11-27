import asyncio
import websockets
import numpy as np
import wave
import io
from datetime import datetime
import json
import os

# 保留原有的模型导入和初始化代码
from transformers import AutoModelForCausalLM, AutoTokenizer
from funasr import AutoModel
import edge_tts

import logging
from logging.handlers import RotatingFileHandler

# 创建日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 创建文件处理器
file_handler = RotatingFileHandler(
    'server.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(formatter)

# 配置根日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 获取应用程序日志记录器
logger = logging.getLogger(__name__)

class AudioServer:
    def __init__(self):
        logger.info("初始化 AudioServer...")
        self.init_models()
        
    def init_models(self):
        logger.info("开始加载模型...")
        # 初始化语音识别模型
        self.model_senceVoice = AutoModel(
            model="/home/jayliu/voice/pretrained_models/SenseVoiceSmall",
            trust_remote_code=True,
        )
        logger.info("语音识别模型加载完成")
        
        # 初始化QWen模型
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "/data/haoyuche/llm/chat_model/qwen/Qwen2___5-14B-Instruct/",
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/data/haoyuche/llm/chat_model/qwen/Qwen2___5-14B-Instruct/"
        )
        logger.info("QWen模型加载完成")

    async def process_audio(self, audio_data):
        try:
            logger.info("="*50)  # 添加分隔线
            logger.info("开始新的音频处理")
            logger.info(f"开始处理音频数据，大小: {len(audio_data)} 字节")
            
            # 保存临时音频文件
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(audio_data)
                
                # 语音识别
                res = self.model_senceVoice.generate(
                    input=wav_buffer.getvalue(),
                    cache={},
                    language="auto",
                    use_itn=False,
                )
                
                prompt = res[0]['text'].split(">")[-1] + "，回答简短一些，保持50字以内！"
                logger.info(f"语音识别结果: {prompt}")

                # 大模型推理
                messages = [
                    {"role": "system", "content": "你叫智产语音助手，是一个职场女性，性格成熟稳重"},
                    {"role": "user", "content": prompt},
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)
                generated_ids = self.llm_model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                )
                
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                logger.info(f"LLM生成回复: {response_text}")

                # 文本转语音
                logger.info("开始文本转语音...")
                communicate = edge_tts.Communicate(response_text, "zh-CN-XiaoyiNeural")
                
                # 创建临时文件
                temp_file = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
                logger.info(f"创建临时文件: {temp_file}")
                
                await communicate.save(temp_file)
                logger.info("音频文件保存成功")
                
                with open(temp_file, 'rb') as f:
                    audio_bytes = f.read()
                
                os.remove(temp_file)
                logger.info(f"音频数据准备完成，大小: {len(audio_bytes)} 字节")

                return {
                    "text": response_text,
                    "audio": audio_bytes
                }

        except Exception as e:
            logger.error(f"处理音频时发生错误: {e}", exc_info=True)
            return {
                "text": "抱歉，处理您的语音时出现了问题。",
                "audio": b''
            }

    async def handle_websocket(self, websocket):
        try:
            logger.info("="*50)  # 添加分隔线
            logger.info("新的WebSocket连接建立")
            async for message in websocket:
                if isinstance(message, bytes):
                    logger.info("收到音频数据，开始处理...")
                    response = await self.process_audio(message)
                    
                    logger.info("发送文本响应...")
                    await websocket.send(json.dumps({
                        "type": "text",
                        "content": response["text"]
                    }))
                    
                    if response["audio"]:
                        logger.info(f"发送音频数据，大小: {len(response['audio'])} 字节")
                        await websocket.send(response["audio"])
                    else:
                        logger.warning("没有音频数据可发送")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("客户端断开连接")
        except Exception as e:
            logger.error(f"WebSocket处理错误: {e}", exc_info=True)

async def main():
    server = AudioServer()
    async with websockets.serve(server.handle_websocket, "0.0.0.0", 8082):
        print("WebSocket server started on ws://0.0.0.0:8082")
        await asyncio.Future()  

if __name__ == "__main__":
    asyncio.run(main())
