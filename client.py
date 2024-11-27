import asyncio
import websockets
import pyaudio
import wave
import json
import pygame
import io
import sys
import signal
from threading import Event
from datetime import datetime

class AudioClient:
    def __init__(self, server_url='ws://172.16.12.1:8082'):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.server_url = server_url
        self.stop_event = Event()
        
        # 初始化音频系统
        try:
            self.p = pyaudio.PyAudio()
            pygame.mixer.init()
            print("音频系统初始化成功")
        except Exception as e:
            print(f"音频系统初始化失败: {e}")
            sys.exit(1)

    def record_audio(self, duration=5):
        """录制固定时长的音频"""
        print(f"\n开始录音 {duration} 秒...")
        
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * duration)):
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        print("录音结束")
        return b''.join(frames)

    def play_audio(self, audio_data):
        """播放音频数据"""
        try:
            print(f"准备播放音频，数据大小: {len(audio_data)} 字节")
            with io.BytesIO(audio_data) as audio_buffer:
                try:
                    pygame.mixer.music.load(audio_buffer)
                    print("音频加载成功")
                    pygame.mixer.music.play()
                    print("开始播放音频")
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    print("音频播放完成")
                except Exception as e:
                    print(f"pygame播放过程中出错: {e}")
        except Exception as e:
            print(f"播放音频失败: {e}")
            import traceback
            print(traceback.format_exc())

    async def chat_session(self):
        """处理一次完整的对话会话"""
        try:
            async with websockets.connect(self.server_url) as websocket:
                print("已连接到服务器")
                
                while not self.stop_event.is_set():
                    # 录制音频
                    print("\n按Enter开始录音(5秒)...")
                    input()
                    audio_data = self.record_audio(5)
                    
                    # 发送音频数据
                    print("发送音频数据...")
                    await websocket.send(audio_data)
                    
                    # 接收响应
                    print("等待服务器响应...")
                    try:
                        # 等待文本响应
                        text_response = await websocket.recv()
                        print(f"收到文本响应类型: {type(text_response)}")
                        response_data = json.loads(text_response)
                        if response_data["type"] == "text":
                            print(f"\nAI回复: {response_data['content']}")
                        
                        # 等待音频响应
                        print("等待音频响应...")
                        audio_response = await websocket.recv()
                        print(f"收到音频响应类型: {type(audio_response)}")
                        if isinstance(audio_response, bytes):
                            print("播放语音回复...")
                            # 保存音频文件以便检查
                            with open(f"received_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3", "wb") as f:
                                f.write(audio_response)
                            self.play_audio(audio_response)
                        else:
                            print(f"收到非字节类型的音频响应: {audio_response[:100]}")
                            
                    except websockets.exceptions.ConnectionClosed:
                        print("服务器连接已断开")
                        break
                    except json.JSONDecodeError as e:
                        print(f"解析响应数据失败: {e}")
                    
                    # 询问是否继续
                    choice = input("\n继续对话? (y/n): ").lower()
                    if choice != 'y':
                        break
                    
        except websockets.exceptions.ConnectionError:
            print("无法连接到服务器")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            print("会话结束")

    def stop(self):
        """停止客户端"""
        self.stop_event.set()
        pygame.mixer.quit()
        self.p.terminate()

def main():
    # 设置服务器地址
    server_url = 'ws://172.16.12.1:8082'  # 替换为实际的服务器地址
    client = AudioClient(server_url)
    
    # 信号处理
    def signal_handler(sig, frame):
        print("\n正在停止客户端...")
        client.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # 运行客户端
        asyncio.run(client.chat_session())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        client.stop()

if __name__ == "__main__":
    main()
